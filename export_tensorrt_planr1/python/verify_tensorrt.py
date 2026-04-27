#!/usr/bin/env python3
"""
Verify TensorRT engines against ONNX models.

This script compares the outputs of TensorRT engines with ONNX Runtime
to ensure numerical consistency.

Usage:
    python verify_tensorrt.py --trt map_encoder.trt --onnx map_encoder.onnx
    python verify_tensorrt.py --all --trt-dir ./engines --onnx-dir ../export_onnx_planr1
"""

import argparse
import os
import sys
import ctypes
from pathlib import Path

import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install tensorrt pycuda")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("Error: ONNX Runtime not installed. Install with: pip install onnxruntime")
    sys.exit(1)


class TRTInference:
    """TensorRT inference wrapper."""
    
    def __init__(self, engine_path: str, plugin_path: str = None):
        """Load TensorRT engine."""
        # Load plugin if provided
        if plugin_path and os.path.exists(plugin_path):
            print(f"Loading plugin: {plugin_path}")
            ctypes.CDLL(plugin_path)
            trt.init_libnvinfer_plugins(trt.Logger(), "planr1")
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        print(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Get input/output info
        self.input_names = []
        self.output_names = []
        self.bindings = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        
        print(f"  Inputs: {self.input_names}")
        print(f"  Outputs: {self.output_names}")
    
    def infer(self, inputs: dict) -> dict:
        """Run inference."""
        # Allocate device memory
        device_inputs = {}
        device_outputs = {}
        
        # Set input shapes and allocate
        for name in self.input_names:
            data = inputs[name]
            self.context.set_input_shape(name, data.shape)
            device_inputs[name] = cuda.mem_alloc(data.nbytes)
            cuda.memcpy_htod(device_inputs[name], data)
            self.context.set_tensor_address(name, int(device_inputs[name]))
        
        # Allocate outputs
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            output = np.empty(shape, dtype=dtype)
            device_outputs[name] = cuda.mem_alloc(output.nbytes)
            self.context.set_tensor_address(name, int(device_outputs[name]))
        
        # Run inference
        self.context.execute_async_v3(cuda.Stream().handle)
        cuda.Context.synchronize()
        
        # Copy outputs back
        outputs = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            output = np.empty(shape, dtype=dtype)
            cuda.memcpy_dtoh(output, device_outputs[name])
            outputs[name] = output
        
        # Free device memory
        for mem in device_inputs.values():
            mem.free()
        for mem in device_outputs.values():
            mem.free()
        
        return outputs


def create_dummy_map_encoder_inputs():
    """Create dummy inputs for map encoder."""
    MAX_POLYLINES = 1200
    MAX_POLYGONS = 145
    MAX_EDGES = 80
    
    return {
        'polyline_position': np.random.randn(MAX_POLYLINES, 2).astype(np.float32),
        'polyline_heading': np.random.randn(MAX_POLYLINES).astype(np.float32),
        'polyline_length': np.abs(np.random.randn(MAX_POLYLINES)).astype(np.float32) + 0.1,
        'polygon_position': np.random.randn(MAX_POLYGONS, 2).astype(np.float32),
        'polygon_heading': np.random.randn(MAX_POLYGONS).astype(np.float32),
        'polygon_speed_limit': np.random.rand(MAX_POLYGONS).astype(np.float32) * 20,
        'polygon_speed_limit_valid': np.ones(MAX_POLYGONS).astype(np.float32),
        'polygon_type': np.zeros(MAX_POLYGONS).astype(np.int64),
        'polygon_traffic_light': np.full(MAX_POLYGONS, 4).astype(np.int64),
        'polygon_on_route': np.zeros(MAX_POLYGONS).astype(np.int64),
        'l2g_edge_index': np.stack([
            np.arange(MAX_POLYLINES),
            np.arange(MAX_POLYLINES) % MAX_POLYGONS
        ]).astype(np.int64),
        'left_edge_index': np.stack([
            np.arange(MAX_EDGES),
            (np.arange(MAX_EDGES) + 1) % MAX_POLYGONS
        ]).astype(np.int64),
        'right_edge_index': np.stack([
            np.arange(MAX_EDGES),
            (np.arange(MAX_EDGES) + 1) % MAX_POLYGONS
        ]).astype(np.int64),
        'incoming_edge_index': np.stack([
            np.arange(MAX_EDGES),
            (np.arange(MAX_EDGES) + 1) % MAX_POLYGONS
        ]).astype(np.int64),
        'outgoing_edge_index': np.stack([
            np.arange(MAX_EDGES),
            (np.arange(MAX_EDGES) + 1) % MAX_POLYGONS
        ]).astype(np.int64),
    }


def create_dummy_step_inputs(polygon_embs: np.ndarray):
    """Create dummy inputs for step model."""
    MAX_AGENTS = 21
    NUM_INTERVALS = 5
    
    # Create edge indices
    num_k2k_t_edges = 100
    num_g2k_edges = 500
    num_k2k_a_edges = 80
    
    return {
        'agent_token': np.zeros((MAX_AGENTS, NUM_INTERVALS), dtype=np.int64),
        'agent_position': np.random.randn(MAX_AGENTS, NUM_INTERVALS, 2).astype(np.float32),
        'agent_heading': np.random.randn(MAX_AGENTS, NUM_INTERVALS).astype(np.float32),
        'agent_velocity': np.random.randn(MAX_AGENTS, NUM_INTERVALS, 2).astype(np.float32),
        'agent_type': np.zeros(MAX_AGENTS, dtype=np.int64),
        'agent_valid_mask': np.ones((MAX_AGENTS, NUM_INTERVALS), dtype=np.float32),
        'polygon_embs': polygon_embs,
        'k2k_t_edge_index': np.random.randint(0, MAX_AGENTS * NUM_INTERVALS, (2, num_k2k_t_edges)).astype(np.int64),
        'k2k_t_edge_attr': np.random.randn(num_k2k_t_edges, 6).astype(np.float32),
        'g2k_edge_index': np.stack([
            np.random.randint(0, 145, num_g2k_edges),
            np.random.randint(0, MAX_AGENTS * NUM_INTERVALS, num_g2k_edges)
        ]).astype(np.int64),
        'g2k_edge_attr': np.random.randn(num_g2k_edges, 6).astype(np.float32),
        'k2k_a_edge_index': np.random.randint(0, MAX_AGENTS * NUM_INTERVALS, (2, num_k2k_a_edges)).astype(np.int64),
        'k2k_a_edge_attr': np.random.randn(num_k2k_a_edges, 5).astype(np.float32),
    }


def compare_outputs(trt_outputs: dict, ort_outputs: list, output_names: list, tolerance: float = 0.01):
    """Compare TensorRT and ONNX Runtime outputs."""
    results = []
    
    for i, name in enumerate(output_names):
        trt_out = trt_outputs[name]
        ort_out = ort_outputs[i]
        
        # Check for NaN
        trt_nan = np.isnan(trt_out).any()
        ort_nan = np.isnan(ort_out).any()
        
        if trt_nan or ort_nan:
            results.append({
                'name': name,
                'status': 'FAIL',
                'reason': f'NaN detected (TRT: {trt_nan}, ORT: {ort_nan})'
            })
            continue
        
        # Compute difference
        diff = np.abs(trt_out.astype(np.float32) - ort_out.astype(np.float32))
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        # Relative difference
        scale = np.abs(ort_out).max() + 1e-8
        rel_diff = max_diff / scale
        
        status = 'PASS' if rel_diff < tolerance else 'WARN' if rel_diff < 0.1 else 'FAIL'
        
        results.append({
            'name': name,
            'status': status,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'rel_diff': rel_diff,
            'shape': trt_out.shape
        })
    
    return results


def verify_model(trt_path: str, onnx_path: str, plugin_path: str = None, num_samples: int = 5):
    """Verify TensorRT engine against ONNX model."""
    print(f"\n{'='*60}")
    print(f"Verifying: {os.path.basename(trt_path)}")
    print(f"{'='*60}")
    
    # Load models
    trt_infer = TRTInference(trt_path, plugin_path)
    ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    output_names = [out.name for out in ort_session.get_outputs()]
    
    # Determine model type
    is_map_encoder = 'map_encoder' in trt_path.lower()
    
    all_results = []
    
    for i in range(num_samples):
        print(f"\nSample {i+1}/{num_samples}:")
        
        # Create inputs
        if is_map_encoder:
            inputs = create_dummy_map_encoder_inputs()
        else:
            # For step model, we need polygon_embs from map encoder
            # Use random for testing
            polygon_embs = np.random.randn(145, 128).astype(np.float32)
            inputs = create_dummy_step_inputs(polygon_embs)
        
        # Run TensorRT
        trt_outputs = trt_infer.infer(inputs)
        
        # Run ONNX Runtime
        ort_outputs = ort_session.run(None, inputs)
        
        # Compare
        results = compare_outputs(trt_outputs, ort_outputs, output_names)
        all_results.extend(results)
        
        for r in results:
            if r['status'] == 'PASS':
                print(f"  ✓ {r['name']}: PASS (max_diff={r.get('max_diff', 'N/A'):.6f})")
            elif r['status'] == 'WARN':
                print(f"  ⚠ {r['name']}: WARN (max_diff={r.get('max_diff', 'N/A'):.6f})")
            else:
                print(f"  ✗ {r['name']}: FAIL ({r.get('reason', 'max_diff=' + str(r.get('max_diff', 'N/A')))})")
    
    # Summary
    passed = sum(1 for r in all_results if r['status'] == 'PASS')
    warned = sum(1 for r in all_results if r['status'] == 'WARN')
    failed = sum(1 for r in all_results if r['status'] == 'FAIL')
    
    print(f"\nSummary: {passed} PASS, {warned} WARN, {failed} FAIL")
    
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Verify TensorRT engines")
    
    parser.add_argument("--trt", type=str, help="Path to TensorRT engine")
    parser.add_argument("--onnx", type=str, help="Path to ONNX model")
    parser.add_argument("--plugin", type=str, help="Path to plugin library")
    parser.add_argument("--samples", type=int, default=5, help="Number of test samples")
    
    parser.add_argument("--all", action="store_true", help="Verify all models")
    parser.add_argument("--trt-dir", type=str, default="./engines", help="TensorRT engines directory")
    parser.add_argument("--onnx-dir", type=str, default="../export_onnx_planr1", help="ONNX models directory")
    
    args = parser.parse_args()
    
    if args.all:
        # Verify both models
        success = True
        
        map_trt = os.path.join(args.trt_dir, "map_encoder_fp16.trt")
        map_onnx = os.path.join(args.onnx_dir, "map_encoder.onnx")
        if os.path.exists(map_trt) and os.path.exists(map_onnx):
            success &= verify_model(map_trt, map_onnx, args.plugin, args.samples)
        
        step_trt = os.path.join(args.trt_dir, "step_fp16.trt")
        step_onnx = os.path.join(args.onnx_dir, "step.onnx")
        if os.path.exists(step_trt) and os.path.exists(step_onnx):
            success &= verify_model(step_trt, step_onnx, args.plugin, args.samples)
        
        sys.exit(0 if success else 1)
    
    elif args.trt and args.onnx:
        success = verify_model(args.trt, args.onnx, args.plugin, args.samples)
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
