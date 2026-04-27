#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export Plan-R1 Model to Split ONNX Files.

This script exports Plan-R1 as TWO separate ONNX files:
    1. map_encoder.onnx - Computes polygon embeddings (run once per scene)
    2. step.onnx - Single step backbone + decoder (run 16 times in loop)

The autoregressive loop and edge computation are handled externally (Python/C++).

Usage:
    python -m export_onnx_planr1.export_split_onnx \
        --ckpt /path/to/checkpoint.ckpt \
        --output-dir ./export_onnx_planr1/

Output:
    - map_encoder.onnx: Takes map features, outputs polygon embeddings
    - step.onnx: Takes polygon embeddings + agent state + edges, outputs logits
    - inference_loop.py: Python reference implementation of external loop
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLANR1_PATH = os.path.join(SCRIPT_DIR, '..', 'Plan-R1')
sys.path.insert(0, PLANR1_PATH)

NUPLAN_PATH = os.path.join(SCRIPT_DIR, '..', 'nuplan', 'nuplan-devkit')
if os.path.exists(NUPLAN_PATH):
    sys.path.insert(0, NUPLAN_PATH)

from .map_encoder_exportable import MapEncoderExportable
from .step_exportable import StepModel, create_step_dummy_inputs
from .decoder_head_exportable import DecoderHeadExportable

# Default paths
DEFAULT_CHECKPOINT_PATH = '/workspace/planner/Plan-R1/ckpts/fine-tuning.ckpt'
DEFAULT_TOKEN_DICT_PATH = '/workspace/planner/Plan-R1/tokens/tokens_1024.pt'
DEFAULT_OUTPUT_DIR = '/workspace/planner/export_onnx_planr1/'


def load_original_model(checkpoint_path: str, token_dict_path: str, device: str = 'cuda'):
    """Load original Plan-R1 model from checkpoint."""
    print(f"Loading token dictionary from: {token_dict_path}")
    token_dict = torch.load(token_dict_path, map_location='cpu')
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract hyperparameters
    hparams = checkpoint.get('hyper_parameters', {})
    print(f"Found hyperparameters: {list(hparams.keys())}")
    
    # Import original PlanR1
    from model.PlanR1 import PlanR1
    
    model_kwargs = {
        'mode': hparams.get('mode', 'pred'),
        'token_dict_path': token_dict_path,
        'num_tokens': hparams.get('num_tokens', 1024),
        'interval': hparams.get('interval', 5),
        'hidden_dim': hparams.get('hidden_dim', 128),
        'num_historical_steps': hparams.get('num_historical_steps', 20),
        'num_future_steps': hparams.get('num_future_steps', 80),
        'agent_radius': hparams.get('agent_radius', 60.0),
        'polygon_radius': hparams.get('polygon_radius', 30.0),
        'num_attn_layers': hparams.get('num_attn_layers', 6),
        'pred_top_k': hparams.get('pred_top_k', 1),
        'plan_top_k': hparams.get('plan_top_k', 1),
        'rollout_top_k': hparams.get('rollout_top_k', 50),
        'num_samples': hparams.get('num_samples', 4),
        'beta': hparams.get('beta', 0.1),
        'scaling_factor': hparams.get('scaling_factor', 0.1),
        'num_hops': hparams.get('num_hops', 4),
        'num_heads': hparams.get('num_heads', 8),
        'dropout': hparams.get('dropout', 0.1),
        'lr': hparams.get('lr', 3e-4),
        'weight_decay': hparams.get('weight_decay', 1e-4),
        'warmup_epochs': hparams.get('warmup_epochs', 4),
        'T_max': hparams.get('T_max', 32),
        'val_visualization': False,
    }
    
    model = PlanR1(**model_kwargs)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    return model, token_dict, hparams


def create_map_encoder_inputs(device: str = 'cuda'):
    """Create dummy inputs for map encoder export.
    
    IMPORTANT: Edge indices use SINK node (last index) for padding to avoid
    corrupting real node outputs during aggregation.
    """
    MAX_POLYLINES = 1200
    MAX_POLYGONS = 145
    MAX_EDGES = 80
    SINK_POLYGON = MAX_POLYGONS - 1
    SINK_POLYLINE = MAX_POLYLINES - 1
    
    # Create realistic dummy edge indices with valid edges + sink padding
    # L2G: first 100 edges are valid (polyline i -> polygon i % MAX_POLYGONS)
    l2g_edge_index = torch.full((2, MAX_POLYLINES), SINK_POLYGON, device=device, dtype=torch.long)
    l2g_edge_index[0] = SINK_POLYLINE  # Source row also uses sink
    num_valid_l2g = 100
    l2g_edge_index[0, :num_valid_l2g] = torch.arange(num_valid_l2g, device=device)
    l2g_edge_index[1, :num_valid_l2g] = torch.arange(num_valid_l2g, device=device) % (MAX_POLYGONS - 1)
    
    # G2G: first 20 edges per type are valid (polygon i -> polygon (i+1) % MAX_POLYGONS)
    def create_g2g_edges(num_valid=20):
        edge = torch.full((2, MAX_EDGES), SINK_POLYGON, device=device, dtype=torch.long)
        edge[0, :num_valid] = torch.arange(num_valid, device=device)
        edge[1, :num_valid] = (torch.arange(num_valid, device=device) + 1) % (MAX_POLYGONS - 1)
        return edge
    
    return {
        'polyline_position': torch.randn(MAX_POLYLINES, 2, device=device),
        'polyline_heading': torch.randn(MAX_POLYLINES, device=device),
        'polyline_length': torch.abs(torch.randn(MAX_POLYLINES, device=device)) + 0.1,
        'polygon_position': torch.randn(MAX_POLYGONS, 2, device=device),
        'polygon_heading': torch.randn(MAX_POLYGONS, device=device),
        'polygon_speed_limit': torch.rand(MAX_POLYGONS, device=device) * 20,
        'polygon_speed_limit_valid': torch.ones(MAX_POLYGONS, device=device),
        'polygon_type': torch.zeros(MAX_POLYGONS, device=device, dtype=torch.long),
        'polygon_traffic_light': torch.full((MAX_POLYGONS,), 4, device=device, dtype=torch.long),
        'polygon_on_route': torch.zeros(MAX_POLYGONS, device=device, dtype=torch.long),
        'l2g_edge_index': l2g_edge_index,
        'left_edge_index': create_g2g_edges(20),
        'right_edge_index': create_g2g_edges(20),
        'incoming_edge_index': create_g2g_edges(20),
        'outgoing_edge_index': create_g2g_edges(20),
    }


def export_map_encoder(
    map_encoder: nn.Module,
    output_path: str,
    device: str = 'cuda',
    opset_version: int = 16,  # Use opset 16 for scatter_add support
):
    """Export MapEncoder to ONNX."""
    print("\n" + "=" * 60)
    print("Exporting MapEncoder to ONNX")
    print("=" * 60)
    
    map_encoder.eval()
    inputs = create_map_encoder_inputs(device)
    
    input_names = list(inputs.keys())
    output_names = ['polygon_embs']
    input_tuple = tuple(inputs[name] for name in input_names)
    
    print(f"Input names: {len(input_names)} tensors")
    print(f"Output path: {output_path}")
    
    try:
        torch.onnx.export(
            map_encoder,
            input_tuple,
            output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            do_constant_folding=False,  # IMPORTANT: Must be False to avoid edge index being folded
            export_params=True,
        )
        print(f"✓ Successfully exported to {output_path}")
        
        # Verify
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_step_model(
    step_model: nn.Module,
    output_path: str,
    device: str = 'cuda',
    opset_version: int = 16,  # Use opset 16 for scatter_add support
):
    """Export StepModel to ONNX."""
    print("\n" + "=" * 60)
    print("Exporting StepModel to ONNX")
    print("=" * 60)
    
    step_model.eval()
    inputs = create_step_dummy_inputs(device)
    
    input_names = list(inputs.keys())
    output_names = ['logits']
    input_tuple = tuple(inputs[name] for name in input_names)
    
    print(f"Input names: {len(input_names)} tensors")
    print(f"Output path: {output_path}")
    
    # Define dynamic axes for variable edge counts
    dynamic_axes = {
        'k2k_t_edge_index': {1: 'num_k2k_t_edges'},
        'k2k_t_edge_attr': {0: 'num_k2k_t_edges'},
        'g2k_edge_index': {1: 'num_g2k_edges'},
        'g2k_edge_attr': {0: 'num_g2k_edges'},
        'k2k_a_edge_index': {1: 'num_k2k_a_edges'},
        'k2k_a_edge_attr': {0: 'num_k2k_a_edges'},
        'agent_token': {1: 'num_intervals'},
    }
    
    try:
        torch.onnx.export(
            step_model,
            input_tuple,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=False,  # IMPORTANT: Must be False to avoid edge index being folded
            export_params=True,
        )
        print(f"✓ Successfully exported to {output_path}")
        
        # Verify
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_inference_loop_script(output_path: str, token_dict_path: str):
    """Generate Python inference loop script."""
    script_content = '''#!/usr/bin/env python3
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
    parser.add_argument('--tokens', type=str, default='TOKEN_DICT_PATH', help='Path to token dictionary')
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
    print("\\nNote: This is a template script. Implement data loading as needed.")


if __name__ == '__main__':
    main()
'''
    
    # Replace TOKEN_DICT_PATH with actual path
    script_content = script_content.replace('TOKEN_DICT_PATH', token_dict_path)
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    print(f"Generated inference loop script: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Export Plan-R1 to Split ONNX')
    parser.add_argument('--ckpt', type=str, default=DEFAULT_CHECKPOINT_PATH,
                       help='Path to checkpoint file')
    parser.add_argument('--tokens', type=str, default=DEFAULT_TOKEN_DICT_PATH,
                       help='Path to token dictionary file')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--opset', type=int, default=16,
                       help='ONNX opset version (default 16 for scatter_add support)')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test inference, do not export')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Plan-R1 Split ONNX Export")
    print("=" * 60)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Token Dict: {args.tokens}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Device: {args.device}")
    
    # Check files
    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint not found: {args.ckpt}")
        return 1
    if not os.path.exists(args.tokens):
        print(f"Error: Token dictionary not found: {args.tokens}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load original model
    print("\n" + "=" * 60)
    print("Loading Original Model")
    print("=" * 60)
    
    try:
        original_model, token_dict, hparams = load_original_model(
            args.ckpt, args.tokens, args.device
        )
        print("✓ Original model loaded")
    except Exception as e:
        print(f"✗ Failed to load original model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create exportable map encoder
    print("\n" + "=" * 60)
    print("Creating Exportable MapEncoder")
    print("=" * 60)
    
    map_encoder = MapEncoderExportable(
        hidden_dim=hparams.get('hidden_dim', 128),
        num_hops=hparams.get('num_hops', 4),
        num_heads=hparams.get('num_heads', 8),
        dropout=hparams.get('dropout', 0.1),
    ).to(args.device)
    
    try:
        map_encoder.load_weights_from(original_model.pred_map_encoder)
        print("✓ MapEncoder weights loaded")
    except Exception as e:
        print(f"✗ Failed to load MapEncoder weights: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create exportable step model
    print("\n" + "=" * 60)
    print("Creating Exportable StepModel")
    print("=" * 60)
    
    step_model = StepModel(
        hidden_dim=hparams.get('hidden_dim', 128),
        num_tokens=hparams.get('num_tokens', 1024),
        num_attn_layers=hparams.get('num_attn_layers', 6),
        num_heads=hparams.get('num_heads', 8),
        dropout=hparams.get('dropout', 0.1),
    ).to(args.device)
    
    try:
        step_model.load_weights_from(
            original_model.pred_backbone,
            original_model.pred_decoder_head,
        )
        print("✓ StepModel weights loaded")
    except Exception as e:
        print(f"✗ Failed to load StepModel weights: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test inference
    print("\n" + "=" * 60)
    print("Testing Inference")
    print("=" * 60)
    
    map_encoder.eval()
    step_model.eval()
    
    with torch.no_grad():
        # Test map encoder
        map_inputs = create_map_encoder_inputs(args.device)
        try:
            polygon_embs = map_encoder(**map_inputs)
            print(f"✓ MapEncoder output shape: {polygon_embs.shape}")
        except Exception as e:
            print(f"✗ MapEncoder inference failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        # Test step model
        step_inputs = create_step_dummy_inputs(args.device)
        step_inputs['polygon_embs'] = polygon_embs
        try:
            logits = step_model(**step_inputs)
            print(f"✓ StepModel output shape: {logits.shape}")
        except Exception as e:
            print(f"✗ StepModel inference failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    if args.test_only:
        print("\n✓ Test completed successfully")
        return 0
    
    # Export models
    map_encoder_path = os.path.join(args.output_dir, 'map_encoder.onnx')
    step_model_path = os.path.join(args.output_dir, 'step.onnx')
    
    # Export map encoder
    if not export_map_encoder(map_encoder, map_encoder_path, args.device, args.opset):
        return 1
    
    # Export step model
    if not export_step_model(step_model, step_model_path, args.device, args.opset):
        return 1
    
    # Generate inference loop script
    inference_script_path = os.path.join(args.output_dir, 'inference_loop.py')
    generate_inference_loop_script(inference_script_path, args.tokens)
    
    # Summary
    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    
    for path in [map_encoder_path, step_model_path, inference_script_path]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {os.path.basename(path)}: {size_mb:.2f} MB")
    
    print(f"\nUsage:")
    print(f"  python {inference_script_path} \\")
    print(f"    --map-encoder {map_encoder_path} \\")
    print(f"    --step {step_model_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
