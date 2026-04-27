#!/usr/bin/env python3
"""
Convert Plan-R1 ONNX models to TensorRT engines with custom ScatterAdd plugin.

This script:
1. Loads the ONNX model
2. Replaces ScatterElements(reduction='add') with custom ScatterAdd nodes
3. Builds TensorRT engine with FP16 support

Usage:
    python convert_to_tensorrt.py --onnx map_encoder.onnx --output map_encoder.trt --fp16
    python convert_to_tensorrt.py --onnx step.onnx --output step.trt --fp16 --dynamic
    python convert_to_tensorrt.py --all --input-dir ../export_onnx_planr1 --output-dir ./engines
"""

import argparse
import os
import sys
import ctypes
from pathlib import Path

import numpy as np

try:
    import tensorrt as trt
except ImportError:
    print("Error: TensorRT not installed. Install with: pip install tensorrt")
    sys.exit(1)

try:
    import onnx
    from onnx import helper, numpy_helper
except ImportError:
    print("Error: ONNX not installed. Install with: pip install onnx")
    sys.exit(1)


# TensorRT Logger
class TRTLogger(trt.Logger):
    def __init__(self, verbosity=trt.Logger.WARNING):
        super().__init__(verbosity)
    
    def log(self, severity, msg):
        if severity <= self.min_severity:
            print(f"[TRT] {msg}")


def load_plugin_library(plugin_path: str):
    """Load custom plugin shared library."""
    if not os.path.exists(plugin_path):
        raise FileNotFoundError(f"Plugin library not found: {plugin_path}")
    
    print(f"Loading plugin library: {plugin_path}")
    ctypes.CDLL(plugin_path)
    
    # Initialize TensorRT plugins
    trt.init_libnvinfer_plugins(TRTLogger(), "")


def modify_onnx_for_plugin(model_path: str, output_path: str = None) -> str:
    """
    Modify ONNX model to use custom ScatterAdd plugin instead of ScatterElements.
    
    The ONNX ScatterElements with reduction='add' is not supported by TensorRT.
    We replace it with a custom 'ScatterAdd' node that our plugin handles.
    """
    print(f"Loading ONNX model: {model_path}")
    model = onnx.load(model_path)
    graph = model.graph
    
    modified = False
    nodes_to_replace = []
    
    # Find all ScatterElements nodes with reduction='add'
    for i, node in enumerate(graph.node):
        if node.op_type == "ScatterElements":
            for attr in node.attribute:
                if attr.name == "reduction" and attr.s == b"add":
                    nodes_to_replace.append((i, node))
                    break
    
    if not nodes_to_replace:
        print("No ScatterElements with reduction='add' found. Model may already be compatible.")
        return model_path
    
    print(f"Found {len(nodes_to_replace)} ScatterElements nodes to replace")
    
    # Replace nodes
    for i, old_node in reversed(nodes_to_replace):
        # ScatterElements inputs: data, indices, updates
        # ScatterAdd inputs: data, index (1D), src
        
        # Get axis attribute
        axis = 0
        for attr in old_node.attribute:
            if attr.name == "axis":
                axis = attr.i
        
        # Create custom ScatterAdd node
        new_node = helper.make_node(
            "ScatterAdd",
            inputs=old_node.input,
            outputs=old_node.output,
            name=f"ScatterAdd_{i}",
            domain="",  # Use default domain
            axis=axis
        )
        
        # Replace in graph
        graph.node.remove(old_node)
        graph.node.insert(i, new_node)
        modified = True
        print(f"  Replaced node {i}: {old_node.name} -> ScatterAdd_{i}")
    
    if modified:
        # No need to add custom domain - using default namespace
        pass
        
        # Save modified model
        if output_path is None:
            base, ext = os.path.splitext(model_path)
            output_path = f"{base}_modified{ext}"
        
        onnx.save(model, output_path)
        print(f"Saved modified model: {output_path}")
        return output_path
    
    return model_path


def build_engine(
    onnx_path: str,
    output_path: str,
    fp16: bool = True,
    dynamic: bool = False,
    max_batch: int = 1,
    max_workspace_mb: int = 4096,
    plugin_path: str = None
):
    """
    Build TensorRT engine from ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save TensorRT engine
        fp16: Enable FP16 mode
        dynamic: Enable dynamic shapes
        max_batch: Maximum batch size
        max_workspace_mb: Maximum workspace size in MB
        plugin_path: Path to custom plugin library
    """
    # Load plugin if provided
    if plugin_path:
        load_plugin_library(plugin_path)
    
    logger = TRTLogger(trt.Logger.INFO)
    
    # Create builder
    builder = trt.Builder(logger)
    
    # Create network
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    # Create parser
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX model!")
            for i in range(parser.num_errors):
                print(f"  Error {i}: {parser.get_error(i)}")
            return False
    
    print(f"Network has {network.num_inputs} inputs and {network.num_outputs} outputs")
    
    # Print input/output info
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  Input {i}: {inp.name} {inp.shape} {inp.dtype}")
    
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  Output {i}: {out.name} {out.shape} {out.dtype}")
    
    # Create builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_mb * 1024 * 1024)
    
    # Enable FP16
    if fp16:
        if builder.platform_has_fast_fp16:
            print("Enabling FP16 mode")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("Warning: Platform does not have fast FP16 support")
    
    # Configure dynamic shapes if needed
    if dynamic:
        print("Configuring dynamic shapes...")
        profile = builder.create_optimization_profile()
        
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            shape = inp.shape
            
            # Check for dynamic dimensions
            has_dynamic = any(d == -1 for d in shape)
            
            if has_dynamic:
                # Set min/opt/max shapes
                min_shape = list(shape)
                opt_shape = list(shape)
                max_shape = list(shape)
                
                for j, d in enumerate(shape):
                    if d == -1:
                        # Dynamic dimension
                        if "edge" in inp.name.lower() or "index" in inp.name.lower():
                            # Edge tensors: can vary from 1 to 10000
                            min_shape[j] = 1
                            opt_shape[j] = 1000
                            max_shape[j] = 10000
                        elif "agent" in inp.name.lower():
                            # Agent tensors
                            min_shape[j] = 1
                            opt_shape[j] = 21
                            max_shape[j] = 21
                        else:
                            # Default
                            min_shape[j] = 1
                            opt_shape[j] = 100
                            max_shape[j] = 1000
                
                print(f"  {inp.name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
                profile.set_shape(
                    inp.name,
                    tuple(min_shape),
                    tuple(opt_shape),
                    tuple(max_shape)
                )
        
        config.add_optimization_profile(profile)
    
    # Build engine
    print("Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("Failed to build engine!")
        return False
    
    # Save engine
    print(f"Saving engine to: {output_path}")
    
    # Handle both bytes and IHostMemory types
    if hasattr(serialized_engine, 'nbytes'):
        engine_bytes = serialized_engine.nbytes
        with open(output_path, 'wb') as f:
            f.write(memoryview(serialized_engine))
    else:
        engine_bytes = len(serialized_engine)
        with open(output_path, 'wb') as f:
            f.write(serialized_engine)
    
    print(f"Engine saved successfully! Size: {engine_bytes / 1024 / 1024:.2f} MB")
    return True


def convert_all(input_dir: str, output_dir: str, fp16: bool = True, plugin_path: str = None):
    """Convert both map_encoder.onnx and step.onnx."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Map encoder (fixed shapes)
    map_encoder_onnx = os.path.join(input_dir, "map_encoder.onnx")
    if os.path.exists(map_encoder_onnx):
        print("\n" + "="*60)
        print("Converting map_encoder.onnx")
        print("="*60)
        
        # Modify ONNX for plugin
        modified_onnx = os.path.join(output_dir, "map_encoder_modified.onnx")
        modified_path = modify_onnx_for_plugin(map_encoder_onnx, modified_onnx)
        
        # Build engine
        output_path = os.path.join(output_dir, "map_encoder_fp16.trt" if fp16 else "map_encoder.trt")
        success = build_engine(
            modified_path, output_path,
            fp16=fp16, dynamic=False,
            plugin_path=plugin_path
        )
        if not success:
            print("Failed to convert map_encoder!")
    else:
        print(f"Warning: {map_encoder_onnx} not found")
    
    # Step model (dynamic shapes for edges)
    step_onnx = os.path.join(input_dir, "step.onnx")
    if os.path.exists(step_onnx):
        print("\n" + "="*60)
        print("Converting step.onnx")
        print("="*60)
        
        # Modify ONNX for plugin
        modified_onnx = os.path.join(output_dir, "step_modified.onnx")
        modified_path = modify_onnx_for_plugin(step_onnx, modified_onnx)
        
        # Build engine with dynamic shapes
        output_path = os.path.join(output_dir, "step_fp16.trt" if fp16 else "step.trt")
        success = build_engine(
            modified_path, output_path,
            fp16=fp16, dynamic=True,
            plugin_path=plugin_path
        )
        if not success:
            print("Failed to convert step!")
    else:
        print(f"Warning: {step_onnx} not found")


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT with custom plugins")
    
    parser.add_argument("--onnx", type=str, help="Path to ONNX model")
    parser.add_argument("--output", type=str, help="Path to output TensorRT engine")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 mode (default: True)")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 mode instead of FP16")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic shapes")
    parser.add_argument("--plugin", type=str, help="Path to custom plugin library (.so)")
    parser.add_argument("--workspace", type=int, default=4096, help="Workspace size in MB")
    
    parser.add_argument("--all", action="store_true", help="Convert all models")
    parser.add_argument("--input-dir", type=str, default="../export_onnx_planr1",
                        help="Input directory containing ONNX files")
    parser.add_argument("--output-dir", type=str, default="./engines",
                        help="Output directory for TensorRT engines")
    
    args = parser.parse_args()
    
    fp16 = not args.fp32
    
    if args.all:
        convert_all(args.input_dir, args.output_dir, fp16=fp16, plugin_path=args.plugin)
    elif args.onnx:
        if not args.output:
            base = os.path.splitext(args.onnx)[0]
            args.output = f"{base}_fp16.trt" if fp16 else f"{base}.trt"
        
        # Modify ONNX for plugin
        modified_path = modify_onnx_for_plugin(args.onnx)
        
        # Build engine
        build_engine(
            modified_path, args.output,
            fp16=fp16, dynamic=args.dynamic,
            max_workspace_mb=args.workspace,
            plugin_path=args.plugin
        )
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python convert_to_tensorrt.py --all")
        print("  python convert_to_tensorrt.py --onnx map_encoder.onnx --output map_encoder.trt")


if __name__ == "__main__":
    main()
