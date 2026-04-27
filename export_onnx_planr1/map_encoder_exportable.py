#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exportable MapEncoder for Plan-R1 ONNX export.

This module provides a fixed-dimension version of MapEncoder that doesn't
depend on PyTorch Geometric's dynamic graph operations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .layers_exportable import (
    TwoLayerMLP,
    GraphAttentionExportable,
    copy_graph_attention_weights,
)


def compute_angles_lengths_2D(vector: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute angles and lengths of 2D vectors.
    
    For zero-length vectors, returns angle 0 to avoid NaN from atan2(0, 0).
    """
    length = torch.norm(vector, dim=-1)
    # Add small epsilon to avoid atan2(0, 0) which can cause NaN in ONNX
    # When length is near zero, angle is undefined anyway, so we default to 0
    safe_x = vector[..., 0] + eps * (length < eps).float()
    theta = torch.atan2(vector[..., 1], safe_x)
    return length, theta


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def transform_to_local(
    point: torch.Tensor,
    origin: torch.Tensor,
    heading: torch.Tensor,
) -> torch.Tensor:
    """Transform points to local coordinate frame."""
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    
    delta = point - origin
    
    local_x = delta[..., 0] * cos_h + delta[..., 1] * sin_h
    local_y = -delta[..., 0] * sin_h + delta[..., 1] * cos_h
    
    return torch.stack([local_x, local_y], dim=-1)


class MapEncoderExportable(nn.Module):
    """
    Exportable MapEncoder without PyG dependencies.
    
    ONNX-compatible version: no dynamic conditional branches.
    All operations use fixed dimensions with masking.
    """
    
    # Fixed dimensions for ONNX export
    MAX_POLYLINES = 1200
    MAX_POLYGONS = 145
    MAX_L2G_EDGES = 1200
    MAX_G2G_EDGES = 80
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_hops: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout

        # Embeddings
        self._polygon_types = ['LANE', 'CROSSWALK', 'DRIVABLE_AREA_SEGMENT', 'STATIC_OBJECT']
        self._polygon_embs = nn.Embedding(len(self._polygon_types), hidden_dim)
        
        self._traffic_light_types = ['GREEN', 'YELLOW', 'RED', 'UNKNOWN', 'NONE']
        self._traffic_light_type_embs = nn.Embedding(len(self._traffic_light_types), hidden_dim)
        
        self._route_types = ['YES', 'NO']
        self._route_embs = nn.Embedding(len(self._route_types), hidden_dim)

        # Feature embedding layers
        self.l_emb_layer = TwoLayerMLP(input_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.g_emb_layer = TwoLayerMLP(input_dim=2, hidden_dim=hidden_dim, output_dim=hidden_dim)

        # Edge embedding layers
        self.l2g_emb_layer = TwoLayerMLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.g2g_emb_layer = TwoLayerMLP(input_dim=8, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        self._g2g_edge_types = ['LEFT', 'RIGHT', 'INCOMING', 'OUTGOING']

        # Attention layers
        self.l2g_attn_layer = GraphAttentionExportable(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            has_edge_attr=True,
            if_self_attention=False,
        )
        self.g2g_attn_layer = GraphAttentionExportable(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            has_edge_attr=True,
            if_self_attention=True,
        )

    def forward(
        self,
        # Polyline features
        polyline_position: torch.Tensor,     # [num_polylines, 2]
        polyline_heading: torch.Tensor,      # [num_polylines]
        polyline_length: torch.Tensor,       # [num_polylines]
        # Polygon features
        polygon_position: torch.Tensor,      # [num_polygons, 2]
        polygon_heading: torch.Tensor,       # [num_polygons]
        polygon_speed_limit: torch.Tensor,   # [num_polygons]
        polygon_speed_limit_valid: torch.Tensor,  # [num_polygons]
        polygon_type: torch.Tensor,          # [num_polygons]
        polygon_traffic_light: torch.Tensor, # [num_polygons]
        polygon_on_route: torch.Tensor,      # [num_polygons]
        # Edge indices (fixed size, padded with zeros)
        l2g_edge_index: torch.Tensor,        # [2, MAX_L2G_EDGES]
        left_edge_index: torch.Tensor,       # [2, MAX_G2G_EDGES]
        right_edge_index: torch.Tensor,      # [2, MAX_G2G_EDGES]
        incoming_edge_index: torch.Tensor,   # [2, MAX_G2G_EDGES]
        outgoing_edge_index: torch.Tensor,   # [2, MAX_G2G_EDGES]
        # Edge masks (which edges are valid)
        n_l2g_edges: int = None,
        n_left_edges: int = None,
        n_right_edges: int = None,
        n_incoming_edges: int = None,
        n_outgoing_edges: int = None,
    ) -> torch.Tensor:
        """
        Forward pass of MapEncoder.
        
        Note: All edge indices are fixed size. Invalid edges (padding) should
        have index 0 and will be handled via masking in attention.
        
        Returns:
            polygon embeddings [num_polygons, hidden_dim]
        """
        device = polyline_position.device
        
        # Use fixed dimensions for ONNX compatibility
        num_polylines = self.MAX_POLYLINES
        num_polygons = self.MAX_POLYGONS
        
        # Polyline embedding
        l_embs = self.l_emb_layer(polyline_length.unsqueeze(-1))  # [MAX_POLYLINES, D]
        
        # Polygon embedding
        g_embs = self.g_emb_layer(
            torch.stack([polygon_speed_limit, polygon_speed_limit_valid.float()], dim=-1)
        )  # [MAX_POLYGONS, D]
        g_embs = g_embs + self._polygon_embs(polygon_type.long())  # [MAX_POLYGONS, D]
        
        # ===== L2G Attention (always compute, use full edges) =====
        # Clamp indices to valid range (handle padding zeros)
        l2g_src = l2g_edge_index[0].clamp(0, num_polylines - 1)
        l2g_dst = l2g_edge_index[1].clamp(0, num_polygons - 1)
        
        l2g_edge_vector = transform_to_local(
            polyline_position[l2g_src],
            polygon_position[l2g_dst],
            polygon_heading[l2g_dst],
        )
        l2g_length, l2g_theta = compute_angles_lengths_2D(l2g_edge_vector)
        l2g_heading_diff = wrap_angle(
            polyline_heading[l2g_src] - polygon_heading[l2g_dst]
        )
        l2g_edge_attr = torch.stack([
            l2g_length,
            torch.cos(l2g_theta),
            torch.sin(l2g_theta),
            torch.cos(l2g_heading_diff),
            torch.sin(l2g_heading_diff),
        ], dim=-1)
        l2g_edge_embs = self.l2g_emb_layer(l2g_edge_attr)
        
        # L2G attention (bipartite: polyline -> polygon)
        g_embs = self.l2g_attn_layer(
            x=(l_embs, g_embs),
            edge_index=torch.stack([l2g_src, l2g_dst], dim=0),
            edge_attr=l2g_edge_embs,
            num_dst=num_polygons,
        )
        
        # ===== G2G Attention =====
        # Concatenate all edge types
        # Left: type=0, hop=1
        # Right: type=1, hop=1
        # Incoming: type=2, hop=1
        # Outgoing: type=3, hop=1
        
        all_edge_index = torch.cat([
            left_edge_index,
            right_edge_index,
            incoming_edge_index,
            outgoing_edge_index,
        ], dim=1)  # [2, total_edges]
        
        total_edges = all_edge_index.shape[1]
        
        # Edge types for each group
        n_left = left_edge_index.shape[1]
        n_right = right_edge_index.shape[1]
        n_incoming = incoming_edge_index.shape[1]
        n_outgoing = outgoing_edge_index.shape[1]
        
        left_type = torch.zeros(n_left, device=device)
        right_type = torch.ones(n_right, device=device)
        incoming_type = torch.full((n_incoming,), 2, device=device)
        outgoing_type = torch.full((n_outgoing,), 3, device=device)
        
        all_edge_type = torch.cat([left_type, right_type, incoming_type, outgoing_type], dim=0).long()
        all_edge_hop = torch.ones(total_edges, device=device)
        
        # Clamp indices
        g2g_src = all_edge_index[0].clamp(0, num_polygons - 1)
        g2g_dst = all_edge_index[1].clamp(0, num_polygons - 1)
        
        # Edge attributes
        g2g_edge_vector = transform_to_local(
            polygon_position[g2g_src],
            polygon_position[g2g_dst],
            polygon_heading[g2g_dst],
        )
        g2g_length, g2g_theta = compute_angles_lengths_2D(g2g_edge_vector)
        g2g_heading_diff = wrap_angle(
            polygon_heading[g2g_src] - polygon_heading[g2g_dst]
        )
        
        # One-hot edge type
        g2g_edge_type_onehot = F.one_hot(all_edge_type, num_classes=4).float()
        
        g2g_edge_attr = torch.cat([
            g2g_length.unsqueeze(-1),
            g2g_theta.unsqueeze(-1),
            g2g_heading_diff.unsqueeze(-1),
            g2g_edge_type_onehot,
            all_edge_hop.unsqueeze(-1),
        ], dim=-1)
        g2g_edge_embs = self.g2g_emb_layer(g2g_edge_attr)
        
        # G2G attention
        g_embs = self.g2g_attn_layer(
            x=g_embs,
            edge_index=torch.stack([g2g_src, g2g_dst], dim=0),
            edge_attr=g2g_edge_embs,
        )
        
        # Add traffic light embeddings
        g_embs = g_embs + self._traffic_light_type_embs(polygon_traffic_light.long())
        
        # Add route embeddings
        g_embs = g_embs + self._route_embs(polygon_on_route.long())
        
        return g_embs

    def load_weights_from(self, src_encoder):
        """
        Load weights from original MapEncoder.
        
        Args:
            src_encoder: Original MapEncoder module
        """
        # Copy embedding weights
        self._polygon_embs.load_state_dict(src_encoder._polygon_embs.state_dict())
        self._traffic_light_type_embs.load_state_dict(src_encoder._traffic_light_type_embs.state_dict())
        self._route_embs.load_state_dict(src_encoder._route_embs.state_dict())
        
        # Copy MLP weights
        self.l_emb_layer.load_state_dict(src_encoder.l_emb_layer.state_dict())
        self.g_emb_layer.load_state_dict(src_encoder.g_emb_layer.state_dict())
        self.l2g_emb_layer.load_state_dict(src_encoder.l2g_emb_layer.state_dict())
        self.g2g_emb_layer.load_state_dict(src_encoder.g2g_emb_layer.state_dict())
        
        # Copy attention weights
        copy_graph_attention_weights(src_encoder.l2g_attn_layer, self.l2g_attn_layer)
        copy_graph_attention_weights(src_encoder.g2g_attn_layer, self.g2g_attn_layer)
