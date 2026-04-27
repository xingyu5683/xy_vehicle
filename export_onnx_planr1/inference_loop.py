#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan-R1 Inference Loop with ONNX Models.

This script demonstrates how to run inference using the split ONNX models:
    1. map_encoder.onnx - Run once per scene
    2. step.onnx - Run 16 times in autoregressive loop

Usage:
    python inference_loop.py --map-encoder map_encoder.onnx --step step.onnx --data input.npz
"""

import numpy as np
import torch
import onnxruntime as ort
from typing import Dict, Tuple
import argparse


# Constants
MAX_AGENTS = 21
MAX_POLYGONS = 145
NUM_FUTURE_INTERVALS = 16
INTERVAL = 5


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def transform_to_local(point, origin, heading):
    """Transform point to local coordinate frame."""
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    delta = point - origin
    local_x = delta[..., 0] * cos_h + delta[..., 1] * sin_h
    local_y = -delta[..., 0] * sin_h + delta[..., 1] * cos_h
    return np.stack([local_x, local_y], axis=-1)


def compute_k2k_t_edges(position, heading, valid_mask, N, T, max_window=6):
    """
    Compute temporal self-attention edges.
    
    Returns:
        edge_index: [2, E] - source and destination indices
        edge_attr: [E, 6] - edge attributes
    """
    src_list = []
    dst_list = []
    attr_list = []
    
    for n in range(N):
        for t_dst in range(T):
            if not valid_mask[n, t_dst]:
                continue
            for t_src in range(max(0, t_dst - max_window), t_dst + 1):
                if not valid_mask[n, t_src]:
                    continue
                
                src_idx = n * T + t_src
                dst_idx = n * T + t_dst
                
                pos_src = position[n, t_src]
                pos_dst = position[n, t_dst]
                heading_src = heading[n, t_src]
                heading_dst = heading[n, t_dst]
                
                edge_vector = transform_to_local(
                    pos_src.reshape(1, 2),
                    pos_dst.reshape(1, 2),
                    np.array([heading_dst]),
                ).squeeze(0)
                
                length = np.linalg.norm(edge_vector)
                theta = np.arctan2(edge_vector[1], edge_vector[0])
                heading_diff = wrap_angle(heading_src - heading_dst)
                interval_diff = float(t_src - t_dst) * INTERVAL
                
                attr = [
                    length,
                    np.cos(theta),
                    np.sin(theta),
                    np.cos(heading_diff),
                    np.sin(heading_diff),
                    interval_diff,
                ]
                
                src_list.append(src_idx)
                dst_list.append(dst_idx)
                attr_list.append(attr)
    
    if len(src_list) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 6), dtype=np.float32)
    
    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr = np.array(attr_list, dtype=np.float32)
    
    return edge_index, edge_attr


def compute_g2k_edges(polygon_position, polygon_heading, polygon_heading_valid,
                      agent_position, agent_heading, agent_valid_mask,
                      M, N, T, radius=30.0):
    """
    Compute map-to-agent cross-attention edges.
    """
    src_list = []
    dst_list = []
    attr_list = []
    
    for n in range(N):
        for t in range(T):
            if not agent_valid_mask[n, t]:
                continue
            
            agent_pos = agent_position[n, t]
            agent_head = agent_heading[n, t]
            dst_idx = n * T + t
            
            for m in range(M):
                poly_pos = polygon_position[m]
                dist = np.linalg.norm(poly_pos - agent_pos)
                
                if dist >= radius:
                    continue
                
                src_idx = m
                
                edge_vector = transform_to_local(
                    poly_pos.reshape(1, 2),
                    agent_pos.reshape(1, 2),
                    np.array([agent_head]),
                ).squeeze(0)
                
                length = np.linalg.norm(edge_vector)
                theta = np.arctan2(edge_vector[1], edge_vector[0])
                heading_diff = wrap_angle(polygon_heading[m] - agent_head)
                heading_valid = float(polygon_heading_valid[m])
                
                attr = [
                    length,
                    np.cos(theta),
                    np.sin(theta),
                    np.cos(heading_diff),
                    np.sin(heading_diff),
                    heading_valid,
                ]
                
                src_list.append(src_idx)
                dst_list.append(dst_idx)
                attr_list.append(attr)
    
    if len(src_list) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 6), dtype=np.float32)
    
    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr = np.array(attr_list, dtype=np.float32)
    
    return edge_index, edge_attr


def compute_k2k_a_edges(position, heading, valid_mask, N, T, radius=60.0):
    """
    Compute agent-to-agent interaction edges.
    Note: edge indices are in [T*N] space (time-major).
    """
    src_list = []
    dst_list = []
    attr_list = []
    
    for t in range(T):
        for n_dst in range(N):
            if not valid_mask[n_dst, t]:
                continue
            
            dst_idx = t * N + n_dst
            dst_pos = position[n_dst, t]
            dst_head = heading[n_dst, t]
            
            for n_src in range(N):
                if not valid_mask[n_src, t]:
                    continue
                
                src_pos = position[n_src, t]
                dist = np.linalg.norm(src_pos - dst_pos)
                
                if dist >= radius:
                    continue
                
                src_idx = t * N + n_src
                
                edge_vector = transform_to_local(
                    src_pos.reshape(1, 2),
                    dst_pos.reshape(1, 2),
                    np.array([dst_head]),
                ).squeeze(0)
                
                length = np.linalg.norm(edge_vector)
                theta = np.arctan2(edge_vector[1], edge_vector[0])
                heading_diff = wrap_angle(heading[n_src, t] - dst_head)
                
                attr = [
                    length,
                    np.cos(theta),
                    np.sin(theta),
                    np.cos(heading_diff),
                    np.sin(heading_diff),
                ]
                
                src_list.append(src_idx)
                dst_list.append(dst_idx)
                attr_list.append(attr)
    
    if len(src_list) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 5), dtype=np.float32)
    
    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr = np.array(attr_list, dtype=np.float32)
    
    return edge_index, edge_attr


def decode_token_to_trajectory(tokens, token_dict, agent_type, current_position, current_heading):
    """Decode motion tokens to trajectory."""
    N = tokens.shape[0]
    delta = np.zeros((N, 3), dtype=np.float32)
    
    for i in range(N):
        if agent_type[i] == 0:  # Vehicle
            delta[i] = token_dict['Vehicle'][tokens[i]]
        elif agent_type[i] == 1:  # Pedestrian
            delta[i] = token_dict['Pedestrian'][tokens[i]]
        else:  # Bicycle
            delta[i] = token_dict['Bicycle'][tokens[i]]
    
    dx_local = delta[:, 0]
    dy_local = delta[:, 1]
    dheading = delta[:, 2]
    
    cos_h = np.cos(current_heading)
    sin_h = np.sin(current_heading)
    
    dx_global = dx_local * cos_h - dy_local * sin_h
    dy_global = dx_local * sin_h + dy_local * cos_h
    
    next_position = current_position + np.stack([dx_global, dy_global], axis=-1)
    next_heading = wrap_angle(current_heading + dheading)
    
    return next_position, next_heading


def run_inference(
    map_encoder_path: str,
    step_model_path: str,
    token_dict: Dict,
    # Map inputs
    polygon_position: np.ndarray,
    polygon_heading: np.ndarray,
    polygon_heading_valid: np.ndarray,
    # Agent inputs
    agent_token: np.ndarray,      # [N, T_hist]
    agent_position: np.ndarray,   # [N, T_hist, 2]
    agent_heading: np.ndarray,    # [N, T_hist]
    agent_valid_mask: np.ndarray, # [N, T_hist]
    agent_type: np.ndarray,       # [N]
    agent_box: np.ndarray,        # [N, 4]
    agent_identity: np.ndarray,   # [N]
    # Map encoder inputs (for running map encoder)
    map_encoder_inputs: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Plan-R1 inference using ONNX models.
    
    Returns:
        positions: [N, 16, 2] - predicted future positions
        headings: [N, 16] - predicted future headings
    """
    # Create ONNX sessions
    map_encoder_session = ort.InferenceSession(map_encoder_path)
    step_session = ort.InferenceSession(step_model_path)
    
    # Run map encoder (once)
    print("Running map encoder...")
    polygon_embs = map_encoder_session.run(
        ['polygon_embs'],
        map_encoder_inputs,
    )[0]
    
    # Initialize inference state
    N = agent_token.shape[0]
    T_hist = agent_token.shape[1]
    
    infer_token = agent_token.copy()
    infer_position = agent_position.copy()
    infer_heading = agent_heading.copy()
    infer_valid_mask = agent_valid_mask.copy()
    
    output_positions = []
    output_headings = []
    
    # Autoregressive loop
    print(f"Running {NUM_FUTURE_INTERVALS} inference steps...")
    for step in range(NUM_FUTURE_INTERVALS):
        T = infer_token.shape[1]
        
        # Compute edges (this is the key part that runs OUTSIDE ONNX)
        k2k_t_edge_index, k2k_t_edge_attr = compute_k2k_t_edges(
            infer_position, infer_heading, infer_valid_mask, N, T
        )
        g2k_edge_index, g2k_edge_attr = compute_g2k_edges(
            polygon_position, polygon_heading, polygon_heading_valid,
            infer_position, infer_heading, infer_valid_mask,
            len(polygon_position), N, T
        )
        k2k_a_edge_index, k2k_a_edge_attr = compute_k2k_a_edges(
            infer_position, infer_heading, infer_valid_mask, N, T
        )
        
        # Pad agent tensors to fixed size
        agent_token_padded = np.zeros((MAX_AGENTS, T), dtype=np.int64)
        agent_token_padded[:N] = infer_token
        
        agent_type_padded = np.zeros(MAX_AGENTS, dtype=np.int64)
        agent_type_padded[:N] = agent_type
        
        agent_box_padded = np.zeros((MAX_AGENTS, 4), dtype=np.float32)
        agent_box_padded[:N] = agent_box
        
        agent_identity_padded = np.zeros(MAX_AGENTS, dtype=np.int64)
        agent_identity_padded[:N] = agent_identity
        
        # Run step model
        step_inputs = {
            'agent_token': agent_token_padded,
            'agent_type': agent_type_padded,
            'agent_box': agent_box_padded,
            'agent_identity': agent_identity_padded,
            'polygon_embs': polygon_embs,
            'k2k_t_edge_index': k2k_t_edge_index,
            'k2k_t_edge_attr': k2k_t_edge_attr,
            'g2k_edge_index': g2k_edge_index,
            'g2k_edge_attr': g2k_edge_attr,
            'k2k_a_edge_index': k2k_a_edge_index,
            'k2k_a_edge_attr': k2k_a_edge_attr,
            'num_agents': np.array([N], dtype=np.int64),
            'num_intervals': np.array([T], dtype=np.int64),
        }
        
        logits = step_session.run(['logits'], step_inputs)[0]  # [MAX_AGENTS, 1024]
        
        # Sample next token (greedy)
        next_token = np.argmax(logits[:N], axis=-1)  # [N]
        
        # Decode to trajectory
        current_position = infer_position[:, -1]
        current_heading = infer_heading[:, -1]
        
        next_position, next_heading = decode_token_to_trajectory(
            next_token, token_dict, agent_type, current_position, current_heading
        )
        
        # Store output
        output_positions.append(next_position)
        output_headings.append(next_heading)
        
        # Update state for next iteration
        infer_token = np.concatenate([infer_token, next_token[:, None]], axis=1)
        infer_position = np.concatenate([infer_position, next_position[:, None]], axis=1)
        infer_heading = np.concatenate([infer_heading, next_heading[:, None]], axis=1)
        infer_valid_mask = np.concatenate([
            infer_valid_mask,
            np.ones((N, 1), dtype=bool)
        ], axis=1)
    
    # Stack outputs
    positions = np.stack(output_positions, axis=1)  # [N, 16, 2]
    headings = np.stack(output_headings, axis=1)    # [N, 16]
    
    return positions, headings


def main():
    parser = argparse.ArgumentParser(description='Plan-R1 ONNX Inference')
    parser.add_argument('--map-encoder', type=str, required=True, help='Path to map_encoder.onnx')
    parser.add_argument('--step', type=str, required=True, help='Path to step.onnx')
    parser.add_argument('--tokens', type=str, default='/workspace/planner/Plan-R1/tokens/tokens_1024.pt', help='Path to token dictionary')
    parser.add_argument('--data', type=str, help='Path to input data (npz)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Plan-R1 ONNX Inference")
    print("=" * 60)
    print(f"Map encoder: {args.map_encoder}")
    print(f"Step model: {args.step}")
    
    # Load token dictionary
    import torch
    token_dict = torch.load(args.tokens, map_location='cpu')
    token_dict = {k: v.numpy() for k, v in token_dict.items()}
    
    # TODO: Load actual data from npz or create dummy data
    print("\nNote: This is a template script. Implement data loading as needed.")


if __name__ == '__main__':
    main()
