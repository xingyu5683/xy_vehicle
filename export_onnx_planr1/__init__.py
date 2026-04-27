# -*- coding: utf-8 -*-
"""
Export Plan-R1 Model to ONNX

This module provides exportable Plan-R1 model components for ONNX deployment.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    External (Python/C++)                     │
    ├─────────────────────────────────────────────────────────────┤
    │  1. Data Preprocessing                                       │
    │     - Filter agents/polygons within distance                 │
    │     - Compute edge indices and attributes (fixed for 16 steps)│
    │                                                              │
    │  2. Run map_encoder.onnx (once)                              │
    │     Input: polyline/polygon features                         │
    │     Output: polygon_embs [145, 128]                          │
    │                                                              │
    │  3. Autoregressive Loop (16 times)                           │
    │     for step in range(16):                                   │
    │       ├─ Run step.onnx                                       │
    │       │   Input: agent_token, polygon_embs, edges            │
    │       │   Output: logits [21, 1024]                          │
    │       ├─ Sample next token (argmax)                          │
    │       ├─ Decode token -> trajectory (position, heading)      │
    │       └─ Update state                                        │
    └─────────────────────────────────────────────────────────────┘

Default dimensions:
    max_agents = 21 (20 neighbors + 1 ego)
    max_polygons = 145
    max_polylines = 1200
    max_edges = 80
    num_steps = 21 (20 historical + 1 current)
    num_hist_intervals = 4
    num_future_intervals = 16
    interval = 5 frames

Usage:
    # Export models:
    python -m export_onnx_planr1.export_split_onnx --ckpt /path/to/ckpt --output-dir ./
    
    # Verify models:
    python -m export_onnx_planr1.verify_split_onnx --dir ./
    
    # Run inference:
    python export_onnx_planr1/inference_loop.py

Exported ONNX files:
    - map_encoder.onnx: Computes polygon embeddings (run once per scene)
    - step.onnx: Single inference step (run 16 times in autoregressive loop)
"""

from .map_encoder_exportable import MapEncoderExportable
from .decoder_head_exportable import DecoderHeadExportable
from .step_exportable import StepModel, BackboneSingleStep, create_step_dummy_inputs
from .layers_exportable import GraphAttentionExportable, GraphAttentionDense, TwoLayerMLP

__all__ = [
    # Core exportable modules
    'MapEncoderExportable',
    'DecoderHeadExportable',
    'StepModel',
    'BackboneSingleStep',
    
    # Layers
    'GraphAttentionExportable',
    'GraphAttentionDense',
    'TwoLayerMLP',
    
    # Utilities
    'create_step_dummy_inputs',
]
