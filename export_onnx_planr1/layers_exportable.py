#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exportable layers for Plan-R1 ONNX export.

These layers are reimplemented without PyTorch Geometric dependencies,
using fixed-dimension tensor operations that are ONNX-compatible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple


def init_weights(m):
    """Initialize weights."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class TwoLayerMLP(nn.Module):
    """Two-layer MLP with LayerNorm and ReLU."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GraphAttentionExportable(nn.Module):
    """
    Exportable Graph Attention layer without PyG MessagePassing.
    
    This version uses dense attention operations instead of sparse message passing,
    which is compatible with ONNX export.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        has_edge_attr: bool,
        if_self_attention: bool,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.has_edge_attr = has_edge_attr
        self.if_self_attention = if_self_attention

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        
        if has_edge_attr:
            self.edge_k = nn.Linear(hidden_dim, hidden_dim)
            self.edge_v = nn.Linear(hidden_dim, hidden_dim)
            
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        self.attn_drop = nn.Dropout(dropout)
        
        if if_self_attention:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
        else:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
            self.mha_prenorm_dst = nn.LayerNorm(hidden_dim)
            
        if has_edge_attr:
            self.mha_prenorm_edge = nn.LayerNorm(hidden_dim)
            
        self.ffn_prenorm = nn.LayerNorm(hidden_dim)
        self.apply(init_weights)

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        num_dst: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with sparse edge_index.
        
        Args:
            x: Node features. If self_attention, single tensor [N, D].
               If cross-attention, tuple (x_src [M, D], x_dst [N, D]).
            edge_index: [2, E] edge indices, edge_index[0] = src, edge_index[1] = dst
            edge_attr: [E, D] edge features (optional)
            num_dst: Number of destination nodes (for cross-attention)
            
        Returns:
            Updated destination node features [N, D]
        """
        if self.if_self_attention:
            x_src = x_dst = self.mha_prenorm_src(x)
            num_dst_nodes = x.shape[0]
        else:
            x_src, x_dst = x
            x_src = self.mha_prenorm_src(x_src)
            x_dst = self.mha_prenorm_dst(x_dst)
            num_dst_nodes = x_dst.shape[0] if num_dst is None else num_dst
            
        if self.has_edge_attr and edge_attr is not None:
            edge_attr = self.mha_prenorm_edge(edge_attr)
            
        # Multi-head attention via sparse operations
        x_dst_updated = x_dst + self._mha_sparse(x_src, x_dst, edge_index, edge_attr, num_dst_nodes)
        
        # FFN
        x_dst_updated = x_dst_updated + self.ffn(self.ffn_prenorm(x_dst_updated))
        
        return x_dst_updated

    def _mha_sparse(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        num_dst_nodes: int,
    ) -> torch.Tensor:
        """
        Sparse multi-head attention using scatter operations.
        
        Args:
            x_src: [M, D] source node features
            x_dst: [N, D] destination node features
            edge_index: [2, E] edges (src_idx, dst_idx)
            edge_attr: [E, D] edge features
            num_dst_nodes: number of destination nodes
            
        Returns:
            [N, D] aggregated features for destination nodes
        """
        src_idx = edge_index[0]  # [E]
        dst_idx = edge_index[1]  # [E]
        
        # Get features for each edge
        x_src_edge = x_src[src_idx]  # [E, D]
        x_dst_edge = x_dst[dst_idx]  # [E, D]
        
        # Compute Q, K, V
        query = self.q(x_dst_edge).view(-1, self.num_heads, self.head_dim)  # [E, H, D/H]
        key = self.k(x_src_edge).view(-1, self.num_heads, self.head_dim)    # [E, H, D/H]
        value = self.v(x_src_edge).view(-1, self.num_heads, self.head_dim)  # [E, H, D/H]
        
        # Add edge features to K, V
        if self.has_edge_attr and edge_attr is not None:
            key = key + self.edge_k(edge_attr).view(-1, self.num_heads, self.head_dim)
            value = value + self.edge_v(edge_attr).view(-1, self.num_heads, self.head_dim)
        
        # Attention scores
        scale = self.head_dim ** 0.5
        attn_scores = (query * key).sum(dim=-1) / scale  # [E, H]
        
        # Softmax over edges with same destination (using scatter)
        attn_weights = self._scatter_softmax(attn_scores, dst_idx, num_dst_nodes)  # [E, H]
        attn_weights = self.attn_drop(attn_weights)
        
        # Weighted aggregation
        weighted_values = value * attn_weights.unsqueeze(-1)  # [E, H, D/H]
        weighted_values = weighted_values.view(-1, self.hidden_dim)  # [E, D]
        
        # Scatter add to destination nodes
        output = torch.zeros(num_dst_nodes, self.hidden_dim, device=x_src.device, dtype=x_src.dtype)
        output = output.scatter_add(0, dst_idx.unsqueeze(-1).expand(-1, self.hidden_dim), weighted_values)
        
        return output

    def _scatter_softmax(
        self,
        src: torch.Tensor,
        index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Compute softmax over groups defined by index.
        TensorRT-compatible version using only scatter_add (no scatter_reduce max).
        
        For numerical stability, we use clamping instead of max subtraction,
        as TensorRT doesn't support ScatterElements with reduction=max.
        
        Args:
            src: [E, H] values to softmax
            index: [E] group indices
            num_nodes: number of groups
            
        Returns:
            [E, H] softmax values
        """
        E, H = src.shape
        
        # Clamp for numerical stability (instead of subtracting per-group max)
        # This is less precise but TensorRT-compatible
        src_clipped = torch.clamp(src, min=-50.0, max=50.0)
        
        # Compute exp
        exp_src = torch.exp(src_clipped)  # [E, H]
        
        # Sum over groups using scatter_add
        expanded_index = index.unsqueeze(-1).expand(-1, H)
        sum_exp = torch.zeros(num_nodes, H, device=src.device, dtype=src.dtype)
        sum_exp = sum_exp.scatter_add(0, expanded_index, exp_src)
        sum_exp_per_edge = sum_exp[index]  # [E, H]
        
        return exp_src / (sum_exp_per_edge + 1e-8)


class GraphAttentionDense(nn.Module):
    """
    Dense version of Graph Attention for fixed-size inputs.
    
    This version uses dense attention matrices, suitable for fixed-size graphs
    where padding is used.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        has_edge_attr: bool,
        if_self_attention: bool,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.has_edge_attr = has_edge_attr
        self.if_self_attention = if_self_attention
        self.scale = self.head_dim ** 0.5

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        
        if has_edge_attr:
            self.edge_k = nn.Linear(hidden_dim, hidden_dim)
            self.edge_v = nn.Linear(hidden_dim, hidden_dim)
            
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        self.attn_drop = nn.Dropout(dropout)
        
        if if_self_attention:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
        else:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
            self.mha_prenorm_dst = nn.LayerNorm(hidden_dim)
            
        if has_edge_attr:
            self.mha_prenorm_edge = nn.LayerNorm(hidden_dim)
            
        self.ffn_prenorm = nn.LayerNorm(hidden_dim)
        self.apply(init_weights)

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        attn_mask: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with dense attention mask.
        
        Args:
            x: If self_attention: [N, D]. If cross-attention: (x_src [M, D], x_dst [N, D])
            attn_mask: [N, M] or [N, N] boolean mask (True = attend)
            edge_attr: [N, M, D] or [N, N, D] edge features (optional)
            
        Returns:
            Updated features [N, D]
        """
        if self.if_self_attention:
            x_src = x_dst = self.mha_prenorm_src(x)
        else:
            x_src, x_dst = x
            x_src = self.mha_prenorm_src(x_src)
            x_dst = self.mha_prenorm_dst(x_dst)
            
        if self.has_edge_attr and edge_attr is not None:
            edge_attr = self.mha_prenorm_edge(edge_attr)
            
        # Multi-head attention
        x_dst_updated = x_dst + self._mha_dense(x_src, x_dst, attn_mask, edge_attr)
        
        # FFN
        x_dst_updated = x_dst_updated + self.ffn(self.ffn_prenorm(x_dst_updated))
        
        return x_dst_updated

    def _mha_dense(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        attn_mask: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Dense multi-head attention.
        
        Args:
            x_src: [M, D] source features
            x_dst: [N, D] destination features
            attn_mask: [N, M] attention mask
            edge_attr: [N, M, D] edge features (optional)
            
        Returns:
            [N, D] output features
        """
        N = x_dst.shape[0]
        M = x_src.shape[0]
        
        # Compute Q, K, V
        Q = self.q(x_dst).view(N, self.num_heads, self.head_dim)  # [N, H, D/H]
        K = self.k(x_src).view(M, self.num_heads, self.head_dim)  # [M, H, D/H]
        V = self.v(x_src).view(M, self.num_heads, self.head_dim)  # [M, H, D/H]
        
        # Add edge features if provided
        if self.has_edge_attr and edge_attr is not None:
            # edge_attr: [N, M, D]
            edge_k = self.edge_k(edge_attr).view(N, M, self.num_heads, self.head_dim)  # [N, M, H, D/H]
            edge_v = self.edge_v(edge_attr).view(N, M, self.num_heads, self.head_dim)  # [N, M, H, D/H]
            
            # K, V need to be expanded for edge features
            K_expanded = K.unsqueeze(0).expand(N, -1, -1, -1) + edge_k  # [N, M, H, D/H]
            V_expanded = V.unsqueeze(0).expand(N, -1, -1, -1) + edge_v  # [N, M, H, D/H]
            
            # Attention scores: [N, M, H]
            attn_scores = (Q.unsqueeze(1) * K_expanded).sum(dim=-1) / self.scale
        else:
            # Standard attention without edge features
            # [N, H, D/H] @ [M, H, D/H].T -> [N, M, H]
            attn_scores = torch.einsum('nhd,mhd->nmh', Q, K) / self.scale
            V_expanded = V.unsqueeze(0).expand(N, -1, -1, -1)  # [N, M, H, D/H]
        
        # Apply mask
        attn_scores = attn_scores.masked_fill(~attn_mask.unsqueeze(-1), float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=1)  # [N, M, H]
        attn_weights = self.attn_drop(attn_weights)
        
        # Weighted sum
        output = torch.einsum('nmh,nmhd->nhd', attn_weights, V_expanded)  # [N, H, D/H]
        output = output.reshape(N, self.hidden_dim)  # [N, D]
        
        return output


def copy_graph_attention_weights(src_layer, dst_layer):
    """
    Copy weights from PyG GraphAttention to exportable version.
    
    Args:
        src_layer: Original PyG GraphAttention layer
        dst_layer: Exportable GraphAttentionExportable layer
    """
    dst_layer.q.load_state_dict(src_layer.q.state_dict())
    dst_layer.k.load_state_dict(src_layer.k.state_dict())
    dst_layer.v.load_state_dict(src_layer.v.state_dict())
    
    if hasattr(src_layer, 'edge_k') and hasattr(dst_layer, 'edge_k'):
        dst_layer.edge_k.load_state_dict(src_layer.edge_k.state_dict())
        dst_layer.edge_v.load_state_dict(src_layer.edge_v.state_dict())
        
    dst_layer.ffn.load_state_dict(src_layer.ffn.state_dict())
    dst_layer.attn_drop.load_state_dict(src_layer.attn_drop.state_dict())
    dst_layer.mha_prenorm_src.load_state_dict(src_layer.mha_prenorm_src.state_dict())
    
    if hasattr(src_layer, 'mha_prenorm_dst') and hasattr(dst_layer, 'mha_prenorm_dst'):
        dst_layer.mha_prenorm_dst.load_state_dict(src_layer.mha_prenorm_dst.state_dict())
        
    if hasattr(src_layer, 'mha_prenorm_edge') and hasattr(dst_layer, 'mha_prenorm_edge'):
        dst_layer.mha_prenorm_edge.load_state_dict(src_layer.mha_prenorm_edge.state_dict())
        
    dst_layer.ffn_prenorm.load_state_dict(src_layer.ffn_prenorm.state_dict())
