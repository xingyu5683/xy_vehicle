#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exportable Decoder Head for Plan-R1 ONNX export.

This module provides the prediction head for generating next token logits.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .layers_exportable import TwoLayerMLP, init_weights


class DecoderHeadExportable(nn.Module):
    """
    Decoder head for predicting next motion token.
    Uses TwoLayerMLP architecture to match original model.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_tokens: int = 1024,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        
        # Match TwoLayerMLP structure from original model
        self.head = TwoLayerMLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_tokens,
        )
        self.apply(init_weights)

    def forward(self, k_embs: torch.Tensor) -> torch.Tensor:
        """
        Predict token logits.
        
        Args:
            k_embs: [N, T, D] or [N, D] agent embeddings
            
        Returns:
            logits: [N, T, num_tokens] or [N, num_tokens]
        """
        return self.head(k_embs)

    def load_weights_from(self, src_head):
        """Load weights from original decoder head (TwoLayerMLP)."""
        self.head.load_state_dict(src_head.state_dict())


def sample_next_token(
    logits: torch.Tensor,  # [N, num_tokens]
    top_k: int = 1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sample next token from logits.
    
    Args:
        logits: [N, num_tokens] prediction logits
        top_k: Number of top tokens to sample from
        temperature: Sampling temperature
        
    Returns:
        tokens: [N] sampled token indices
    """
    if top_k == 1:
        # Greedy decoding
        return logits.argmax(dim=-1)
    else:
        # Top-k sampling
        logits = logits / temperature
        
        # Get top-k
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
        
        # Sample from top-k
        probs = torch.softmax(top_k_logits, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # Get actual token index
        tokens = top_k_indices.gather(dim=-1, index=sampled_idx.unsqueeze(-1)).squeeze(-1)
        
        return tokens


def decode_token_to_trajectory(
    tokens: torch.Tensor,        # [N] or [N, T] token indices
    token_dict: Dict[str, torch.Tensor],
    agent_type: torch.Tensor,    # [N]
    current_position: torch.Tensor,  # [N, 2]
    current_heading: torch.Tensor,   # [N]
    interval: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decode motion tokens to trajectory.
    
    Args:
        tokens: Token indices
        token_dict: Dictionary with 'veh', 'ped', 'cyc' token centers
        agent_type: Agent types (0=vehicle, 1=pedestrian, 2=bicycle)
        current_position: Current positions [N, 2]
        current_heading: Current headings [N]
        interval: Frame interval
        
    Returns:
        next_position: [N, 2] next positions
        next_heading: [N] next headings
    """
    device = tokens.device
    N = tokens.shape[0]
    
    # Get token centers for each agent type
    # Support both naming conventions
    veh_key = 'Vehicle' if 'Vehicle' in token_dict else 'veh'
    ped_key = 'Pedestrian' if 'Pedestrian' in token_dict else 'ped'
    cyc_key = 'Bicycle' if 'Bicycle' in token_dict else 'cyc'
    
    veh_tokens = token_dict[veh_key].to(device)  # [1024, 3] (dx, dy, dheading)
    ped_tokens = token_dict[ped_key].to(device)  # [1024, 3]
    cyc_tokens = token_dict[cyc_key].to(device)  # [1024, 3]
    
    # Get deltas for each token
    delta = torch.zeros(N, 3, device=device)
    
    vehicle_mask = agent_type == 0
    pedestrian_mask = agent_type == 1
    bicycle_mask = agent_type == 2
    
    if vehicle_mask.any():
        delta[vehicle_mask] = veh_tokens[tokens[vehicle_mask]]
    if pedestrian_mask.any():
        delta[pedestrian_mask] = ped_tokens[tokens[pedestrian_mask]]
    if bicycle_mask.any():
        delta[bicycle_mask] = cyc_tokens[tokens[bicycle_mask]]
    
    # Delta is in local coordinates, transform to global
    dx_local = delta[:, 0]
    dy_local = delta[:, 1]
    dheading = delta[:, 2]
    
    cos_h = torch.cos(current_heading)
    sin_h = torch.sin(current_heading)
    
    dx_global = dx_local * cos_h - dy_local * sin_h
    dy_global = dx_local * sin_h + dy_local * cos_h
    
    next_position = current_position + torch.stack([dx_global, dy_global], dim=-1)
    next_heading = current_heading + dheading
    
    # Wrap heading to [-pi, pi]
    next_heading = torch.atan2(torch.sin(next_heading), torch.cos(next_heading))
    
    return next_position, next_heading
