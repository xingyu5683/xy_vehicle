#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-Step Exportable Model for Plan-R1 ONNX Export.

This module provides a single-step version of the backbone + decoder,
designed for ONNX export. The autoregressive loop is handled externally.

Architecture:
    1. map_encoder.onnx: Computes polygon embeddings (run once)
    2. step.onnx: Single step of backbone + decoder (run 16 times in loop)

Key Design Decisions:
    - Edge indices and attributes are PRE-COMPUTED externally (before ONNX inference)
    - Graph structure remains CONSTANT for all 16 inference steps
    - No Python loops or conditional branches inside ONNX model
    - Fixed tensor dimensions with padding for ONNX compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .layers_exportable import TwoLayerMLP, GraphAttentionExportable, init_weights


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class BackboneSingleStep(nn.Module):
    """
    Single-step backbone for ONNX export.
    
    This processes one inference step, taking pre-computed edges as input.
    No dynamic edge computation inside the model.
    """
    
    # Fixed dimensions
    MAX_AGENTS = 21
    MAX_POLYGONS = 145
    MAX_INTERVALS = 20  # 4 historical + 16 future
    MAX_K2K_T_EDGES = 2000  # Conservative estimate
    MAX_G2K_EDGES = 5000    # Conservative estimate
    MAX_K2K_A_EDGES = 2000  # Conservative estimate
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_tokens: int = 1024,
        num_attn_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.num_attn_layers = num_attn_layers
        self.num_heads = num_heads
        
        # Agent static feature embedding
        self._agent_type_embs = nn.Embedding(3, hidden_dim)  # VEHICLE, PEDESTRIAN, BICYCLE
        self._identity_type_embs = nn.Embedding(2, hidden_dim)  # EGO, AGENT
        self.agent_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        # Token embeddings (one per agent type)
        self.token_emb_vehicle = nn.Embedding(num_tokens, hidden_dim)
        self.token_emb_pedestrian = nn.Embedding(num_tokens, hidden_dim)
        self.token_emb_bicycle = nn.Embedding(num_tokens, hidden_dim)
        
        # Fusion layer
        self.fusion_layer = TwoLayerMLP(input_dim=hidden_dim * 2, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        # Edge embedding layers
        self.k2k_t_emb_layer = TwoLayerMLP(input_dim=6, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.g2k_emb_layer = TwoLayerMLP(input_dim=6, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.k2k_a_emb_layer = TwoLayerMLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        # Attention layers
        self.k2k_t_attn_layers = nn.ModuleList([
            GraphAttentionExportable(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                has_edge_attr=True,
                if_self_attention=True,
            ) for _ in range(num_attn_layers)
        ])
        
        self.g2k_attn_layers = nn.ModuleList([
            GraphAttentionExportable(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                has_edge_attr=True,
                if_self_attention=False,
            ) for _ in range(num_attn_layers)
        ])
        
        self.k2k_a_attn_layers = nn.ModuleList([
            GraphAttentionExportable(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                has_edge_attr=True,
                if_self_attention=True,
            ) for _ in range(num_attn_layers)
        ])
        
        self.apply(init_weights)

    def forward(
        self,
        # Agent token embeddings from previous step (or initial)
        k_embs: torch.Tensor,                   # [N, T, D] - current k embeddings
        # Polygon embeddings (from map encoder)
        polygon_embs: torch.Tensor,             # [M, D] - pre-computed polygon embeddings
        # Pre-computed edge indices (padded to fixed size)
        k2k_t_edge_index: torch.Tensor,         # [2, MAX_K2K_T_EDGES]
        k2k_t_edge_attr: torch.Tensor,          # [MAX_K2K_T_EDGES, 6]
        g2k_edge_index: torch.Tensor,           # [2, MAX_G2K_EDGES]
        g2k_edge_attr: torch.Tensor,            # [MAX_G2K_EDGES, 6]
        k2k_a_edge_index: torch.Tensor,         # [2, MAX_K2K_A_EDGES]
        k2k_a_edge_attr: torch.Tensor,          # [MAX_K2K_A_EDGES, 5]
        # Actual edge counts (for masking padding)
        num_k2k_t_edges: torch.Tensor,          # [1]
        num_g2k_edges: torch.Tensor,            # [1]
        num_k2k_a_edges: torch.Tensor,          # [1]
        # Dimensions
        num_agents: torch.Tensor,               # [1]
        num_intervals: torch.Tensor,            # [1]
    ) -> torch.Tensor:
        """
        Single step forward pass.
        
        Args:
            k_embs: [N, T, D] current agent token embeddings
            polygon_embs: [M, D] pre-computed polygon embeddings
            *_edge_index: Pre-computed edge indices
            *_edge_attr: Pre-computed edge attributes
            
        Returns:
            k_embs_updated: [N, T, D] updated agent token embeddings
        """
        device = k_embs.device
        N = num_agents.item() if isinstance(num_agents, torch.Tensor) else num_agents
        T = num_intervals.item() if isinstance(num_intervals, torch.Tensor) else num_intervals
        M = polygon_embs.shape[0]
        
        # Compute edge embeddings
        n_k2k_t = num_k2k_t_edges.item() if isinstance(num_k2k_t_edges, torch.Tensor) else num_k2k_t_edges
        n_g2k = num_g2k_edges.item() if isinstance(num_g2k_edges, torch.Tensor) else num_g2k_edges
        n_k2k_a = num_k2k_a_edges.item() if isinstance(num_k2k_a_edges, torch.Tensor) else num_k2k_a_edges
        
        # Process valid edges only
        k2k_t_edge_embs = self.k2k_t_emb_layer(k2k_t_edge_attr[:n_k2k_t]) if n_k2k_t > 0 else None
        g2k_edge_embs = self.g2k_emb_layer(g2k_edge_attr[:n_g2k]) if n_g2k > 0 else None
        k2k_a_edge_embs = self.k2k_a_emb_layer(k2k_a_edge_attr[:n_k2k_a]) if n_k2k_a > 0 else None
        
        # Flatten k_embs for attention: [N*T, D]
        k_embs_flat = k_embs[:N, :T].reshape(N * T, self.hidden_dim)
        
        # Attention layers
        for i in range(self.num_attn_layers):
            # k2k_t: temporal attention within each agent
            if n_k2k_t > 0:
                k_embs_flat = self.k2k_t_attn_layers[i](
                    x=k_embs_flat,
                    edge_index=k2k_t_edge_index[:, :n_k2k_t],
                    edge_attr=k2k_t_edge_embs,
                )
            
            # g2k: map-agent cross-attention
            if n_g2k > 0:
                k_embs_flat = self.g2k_attn_layers[i](
                    x=(polygon_embs, k_embs_flat),
                    edge_index=g2k_edge_index[:, :n_g2k],
                    edge_attr=g2k_edge_embs,
                    num_dst=N * T,
                )
            
            # k2k_a: agent-agent interaction
            # Reshape to [T*N, D] for per-timestep attention
            k_embs_flat = k_embs_flat.reshape(N, T, self.hidden_dim).transpose(0, 1).reshape(T * N, self.hidden_dim)
            if n_k2k_a > 0:
                k_embs_flat = self.k2k_a_attn_layers[i](
                    x=k_embs_flat,
                    edge_index=k2k_a_edge_index[:, :n_k2k_a],
                    edge_attr=k2k_a_edge_embs,
                )
            # Reshape back to [N*T, D]
            k_embs_flat = k_embs_flat.reshape(T, N, self.hidden_dim).transpose(0, 1).reshape(N * T, self.hidden_dim)
        
        # Reshape to [N, T, D]
        k_embs_updated = k_embs_flat.reshape(N, T, self.hidden_dim)
        
        # Pad back to original size if needed
        if N < self.MAX_AGENTS or T < self.MAX_INTERVALS:
            k_embs_full = k_embs.clone()
            k_embs_full[:N, :T] = k_embs_updated
            return k_embs_full
        
        return k_embs_updated

    def load_weights_from(self, src_backbone):
        """Load weights from original Backbone."""
        from .layers_exportable import copy_graph_attention_weights
        
        # Embeddings
        self._agent_type_embs.load_state_dict(src_backbone._agent_type_embs.state_dict())
        self._identity_type_embs.load_state_dict(src_backbone._identity_type_embs.state_dict())
        self.agent_emb_layer.load_state_dict(src_backbone.agent_emb_layer.state_dict())
        
        # Token embeddings
        self.token_emb_vehicle.load_state_dict(src_backbone.token_emb_vehicle.state_dict())
        self.token_emb_pedestrian.load_state_dict(src_backbone.token_emb_pedestrian.state_dict())
        self.token_emb_bicycle.load_state_dict(src_backbone.token_emb_bicycle.state_dict())
        
        # Fusion
        self.fusion_layer.load_state_dict(src_backbone.fusion_layer.state_dict())
        
        # Edge embedding
        self.k2k_t_emb_layer.load_state_dict(src_backbone.k2k_t_emb_layer.state_dict())
        self.g2k_emb_layer.load_state_dict(src_backbone.g2k_emb_layer.state_dict())
        self.k2k_a_emb_layer.load_state_dict(src_backbone.k2k_a_emb_layer.state_dict())
        
        # Attention layers
        for i in range(self.num_attn_layers):
            copy_graph_attention_weights(src_backbone.k2k_t_attn_layers[i], self.k2k_t_attn_layers[i])
            copy_graph_attention_weights(src_backbone.g2k_attn_layers[i], self.g2k_attn_layers[i])
            copy_graph_attention_weights(src_backbone.k2k_a_attn_layers[i], self.k2k_a_attn_layers[i])


class StepModel(nn.Module):
    """
    Complete single-step model for ONNX export.
    
    Combines backbone attention + decoder head.
    Takes pre-computed edges and outputs token logits.
    """
    
    MAX_AGENTS = 21
    MAX_POLYGONS = 145
    MAX_INTERVALS = 20
    MAX_K2K_T_EDGES = 2000
    MAX_G2K_EDGES = 5000
    MAX_K2K_A_EDGES = 2000
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_tokens: int = 1024,
        num_attn_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.num_attn_layers = num_attn_layers
        
        # Agent static embeddings
        self._agent_type_embs = nn.Embedding(3, hidden_dim)
        self._identity_type_embs = nn.Embedding(2, hidden_dim)
        self.agent_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        # Token embeddings
        self.token_emb_vehicle = nn.Embedding(num_tokens, hidden_dim)
        self.token_emb_pedestrian = nn.Embedding(num_tokens, hidden_dim)
        self.token_emb_bicycle = nn.Embedding(num_tokens, hidden_dim)
        
        # Fusion layer
        self.fusion_layer = TwoLayerMLP(input_dim=hidden_dim * 2, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        # Edge embedding layers
        self.k2k_t_emb_layer = TwoLayerMLP(input_dim=6, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.g2k_emb_layer = TwoLayerMLP(input_dim=6, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.k2k_a_emb_layer = TwoLayerMLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        # Attention layers
        self.k2k_t_attn_layers = nn.ModuleList([
            GraphAttentionExportable(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                has_edge_attr=True,
                if_self_attention=True,
            ) for _ in range(num_attn_layers)
        ])
        
        self.g2k_attn_layers = nn.ModuleList([
            GraphAttentionExportable(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                has_edge_attr=True,
                if_self_attention=False,
            ) for _ in range(num_attn_layers)
        ])
        
        self.k2k_a_attn_layers = nn.ModuleList([
            GraphAttentionExportable(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                has_edge_attr=True,
                if_self_attention=True,
            ) for _ in range(num_attn_layers)
        ])
        
        # Decoder head
        self.decoder_head = TwoLayerMLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_tokens,
        )
        
        self.apply(init_weights)

    def forward(
        self,
        # Agent features
        agent_token: torch.Tensor,              # [N, T] - token indices for all intervals
        agent_type: torch.Tensor,               # [N] - agent type (0=veh, 1=ped, 2=cyc)
        agent_box: torch.Tensor,                # [N, 4] - bounding box
        agent_identity: torch.Tensor,           # [N] - identity (0=ego, 1=agent)
        # Polygon embeddings (from map encoder)
        polygon_embs: torch.Tensor,             # [M, D] - pre-computed
        # Pre-computed edge indices
        k2k_t_edge_index: torch.Tensor,         # [2, E1]
        k2k_t_edge_attr: torch.Tensor,          # [E1, 6]
        g2k_edge_index: torch.Tensor,           # [2, E2]
        g2k_edge_attr: torch.Tensor,            # [E2, 6]
        k2k_a_edge_index: torch.Tensor,         # [2, E3]
        k2k_a_edge_attr: torch.Tensor,          # [E3, 5]
        # Dimensions
        num_agents: torch.Tensor,               # [1]
        num_intervals: torch.Tensor,            # [1]
    ) -> torch.Tensor:
        """
        Single step forward pass.
        
        Returns:
            logits: [N, num_tokens] - token prediction logits for last timestep
        """
        device = agent_token.device
        N = self.MAX_AGENTS  # Use fixed size for ONNX
        T = num_intervals.item() if isinstance(num_intervals, torch.Tensor) else num_intervals
        M = self.MAX_POLYGONS
        
        # Agent static embedding
        a_embs = (
            self.agent_emb_layer(agent_box) +
            self._agent_type_embs(agent_type.long()) +
            self._identity_type_embs(agent_identity.long())
        )  # [N, D]
        
        # Get token embeddings based on agent type
        k_token_embs = self._get_token_embeddings(agent_token, agent_type, T)  # [N, T, D]
        
        # Fuse static and token embeddings
        a_embs_expanded = a_embs.unsqueeze(1).expand(-1, T, -1)  # [N, T, D]
        k_embs = self.fusion_layer(torch.cat([a_embs_expanded, k_token_embs], dim=-1))  # [N, T, D]
        
        # Compute edge embeddings
        n_k2k_t = k2k_t_edge_index.shape[1]
        n_g2k = g2k_edge_index.shape[1]
        n_k2k_a = k2k_a_edge_index.shape[1]
        
        k2k_t_edge_embs = self.k2k_t_emb_layer(k2k_t_edge_attr) if n_k2k_t > 0 else None
        g2k_edge_embs = self.g2k_emb_layer(g2k_edge_attr) if n_g2k > 0 else None
        k2k_a_edge_embs = self.k2k_a_emb_layer(k2k_a_edge_attr) if n_k2k_a > 0 else None
        
        # Flatten for attention: [N*T, D]
        k_embs_flat = k_embs.reshape(N * T, self.hidden_dim)
        
        # Attention layers
        for i in range(self.num_attn_layers):
            # k2k_t: temporal attention
            if n_k2k_t > 0:
                k_embs_flat = self.k2k_t_attn_layers[i](
                    x=k_embs_flat,
                    edge_index=k2k_t_edge_index,
                    edge_attr=k2k_t_edge_embs,
                )
            
            # g2k: map-agent cross-attention
            if n_g2k > 0:
                k_embs_flat = self.g2k_attn_layers[i](
                    x=(polygon_embs, k_embs_flat),
                    edge_index=g2k_edge_index,
                    edge_attr=g2k_edge_embs,
                    num_dst=N * T,
                )
            
            # k2k_a: agent interaction
            k_embs_flat = k_embs_flat.reshape(N, T, self.hidden_dim).transpose(0, 1).reshape(T * N, self.hidden_dim)
            if n_k2k_a > 0:
                k_embs_flat = self.k2k_a_attn_layers[i](
                    x=k_embs_flat,
                    edge_index=k2k_a_edge_index,
                    edge_attr=k2k_a_edge_embs,
                )
            k_embs_flat = k_embs_flat.reshape(T, N, self.hidden_dim).transpose(0, 1).reshape(N * T, self.hidden_dim)
        
        # Reshape and get last timestep
        k_embs = k_embs_flat.reshape(N, T, self.hidden_dim)
        k_embs_last = k_embs[:, -1]  # [N, D]
        
        # Predict logits
        logits = self.decoder_head(k_embs_last)  # [N, num_tokens]
        
        return logits

    def _get_token_embeddings(
        self,
        tokens: torch.Tensor,  # [N, T]
        agent_type: torch.Tensor,  # [N]
        T: int,
    ) -> torch.Tensor:
        """Get token embeddings based on agent type.
        
        ONNX-compatible version that avoids runtime conditionals.
        Instead of if-else, we compute all embeddings and use masks to select.
        """
        N = tokens.shape[0]
        device = tokens.device
        dtype = self.token_emb_vehicle.weight.dtype
        
        # Compute all embeddings (this is slightly wasteful but ONNX-compatible)
        # Each agent only uses one embedding table based on its type
        
        # Vehicle embeddings for all tokens
        vehicle_embs = self.token_emb_vehicle(tokens)  # [N, T, D]
        
        # Pedestrian embeddings for all tokens
        ped_embs = self.token_emb_pedestrian(tokens)  # [N, T, D]
        
        # Bicycle embeddings for all tokens
        cyc_embs = self.token_emb_bicycle(tokens)  # [N, T, D]
        
        # Create masks for selection [N, 1, 1] for broadcasting
        vehicle_mask = (agent_type == 0).unsqueeze(-1).unsqueeze(-1).float()  # [N, 1, 1]
        ped_mask = (agent_type == 1).unsqueeze(-1).unsqueeze(-1).float()  # [N, 1, 1]
        cyc_mask = (agent_type == 2).unsqueeze(-1).unsqueeze(-1).float()  # [N, 1, 1]
        
        # Weighted sum (only one mask is 1 for each agent)
        result = vehicle_embs * vehicle_mask + ped_embs * ped_mask + cyc_embs * cyc_mask
        
        return result

    def load_weights_from(self, src_backbone, src_decoder_head):
        """Load weights from original backbone and decoder head."""
        from .layers_exportable import copy_graph_attention_weights
        
        # Embeddings
        self._agent_type_embs.load_state_dict(src_backbone._agent_type_embs.state_dict())
        self._identity_type_embs.load_state_dict(src_backbone._identity_type_embs.state_dict())
        self.agent_emb_layer.load_state_dict(src_backbone.agent_emb_layer.state_dict())
        
        # Token embeddings
        self.token_emb_vehicle.load_state_dict(src_backbone.token_emb_vehicle.state_dict())
        self.token_emb_pedestrian.load_state_dict(src_backbone.token_emb_pedestrian.state_dict())
        self.token_emb_bicycle.load_state_dict(src_backbone.token_emb_bicycle.state_dict())
        
        # Fusion
        self.fusion_layer.load_state_dict(src_backbone.fusion_layer.state_dict())
        
        # Edge embedding
        self.k2k_t_emb_layer.load_state_dict(src_backbone.k2k_t_emb_layer.state_dict())
        self.g2k_emb_layer.load_state_dict(src_backbone.g2k_emb_layer.state_dict())
        self.k2k_a_emb_layer.load_state_dict(src_backbone.k2k_a_emb_layer.state_dict())
        
        # Attention layers
        for i in range(self.num_attn_layers):
            copy_graph_attention_weights(src_backbone.k2k_t_attn_layers[i], self.k2k_t_attn_layers[i])
            copy_graph_attention_weights(src_backbone.g2k_attn_layers[i], self.g2k_attn_layers[i])
            copy_graph_attention_weights(src_backbone.k2k_a_attn_layers[i], self.k2k_a_attn_layers[i])
        
        # Decoder head
        self.decoder_head.load_state_dict(src_decoder_head.state_dict())


def create_step_dummy_inputs(device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """Create dummy inputs for step model ONNX export."""
    
    N = StepModel.MAX_AGENTS
    M = StepModel.MAX_POLYGONS
    T = 4  # Start with 4 historical intervals
    D = 128
    
    # Dummy edge counts
    n_k2k_t = 100
    n_g2k = 200
    n_k2k_a = 80
    
    return {
        'agent_token': torch.randint(0, 1024, (N, T), device=device, dtype=torch.long),
        'agent_type': torch.zeros(N, device=device, dtype=torch.long),
        'agent_box': torch.abs(torch.randn(N, 4, device=device)) + 1.0,
        'agent_identity': torch.zeros(N, device=device, dtype=torch.long),
        'polygon_embs': torch.randn(M, D, device=device),
        'k2k_t_edge_index': torch.randint(0, N * T, (2, n_k2k_t), device=device, dtype=torch.long),
        'k2k_t_edge_attr': torch.randn(n_k2k_t, 6, device=device),
        'g2k_edge_index': torch.stack([
            torch.randint(0, M, (n_g2k,), device=device, dtype=torch.long),
            torch.randint(0, N * T, (n_g2k,), device=device, dtype=torch.long),
        ], dim=0),
        'g2k_edge_attr': torch.randn(n_g2k, 6, device=device),
        'k2k_a_edge_index': torch.randint(0, T * N, (2, n_k2k_a), device=device, dtype=torch.long),
        'k2k_a_edge_attr': torch.randn(n_k2k_a, 5, device=device),
        'num_agents': torch.tensor([N], device=device, dtype=torch.long),
        'num_intervals': torch.tensor([T], device=device, dtype=torch.long),
    }
