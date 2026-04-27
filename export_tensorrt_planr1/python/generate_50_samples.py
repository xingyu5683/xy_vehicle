#!/usr/bin/env python3
"""
生成 50 个真实数据样本用于 TensorRT 验证。
"""
import os
import sys
import glob
import torch
import numpy as np

# 设置路径
sys.path.insert(0, '/workspace/planner/Plan-R1')

def main():
    print('=' * 60)
    print('StepModel 真实数据验证 - 50个样本')
    print('=' * 60)

    # 动态导入
    os.chdir('/workspace/planner/export_onnx_planr1')
    sys.path.insert(0, '/workspace/planner/export_onnx_planr1')
    
    # 导入模块
    import importlib.util
    
    # 加载 layers_exportable
    spec = importlib.util.spec_from_file_location("layers_exportable", 
        "/workspace/planner/export_onnx_planr1/layers_exportable.py")
    layers_exportable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layers_exportable)
    
    # 加载 map_encoder_exportable
    spec = importlib.util.spec_from_file_location("map_encoder_exportable", 
        "/workspace/planner/export_onnx_planr1/map_encoder_exportable.py")
    map_encoder_exportable = importlib.util.module_from_spec(spec)
    sys.modules['layers_exportable'] = layers_exportable
    spec.loader.exec_module(map_encoder_exportable)
    
    # 加载 step_exportable  
    spec = importlib.util.spec_from_file_location("step_exportable", 
        "/workspace/planner/export_onnx_planr1/step_exportable.py")
    step_exportable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step_exportable)
    
    # 加载 export_split_onnx
    spec = importlib.util.spec_from_file_location("export_split_onnx", 
        "/workspace/planner/export_onnx_planr1/export_split_onnx.py")
    export_split_onnx = importlib.util.module_from_spec(spec)
    sys.modules['map_encoder_exportable'] = map_encoder_exportable
    sys.modules['step_exportable'] = step_exportable
    spec.loader.exec_module(export_split_onnx)
    
    MapEncoderExportable = map_encoder_exportable.MapEncoderExportable
    StepModel = step_exportable.StepModel
    build_step_edges = step_exportable.build_step_edges
    copy_map_encoder_weights = export_split_onnx.copy_map_encoder_weights
    copy_step_model_weights = export_split_onnx.copy_step_model_weights

    # 加载 PyTorch 模型
    print('\n加载 PyTorch 模型...')
    ckpt_path = '/workspace/planner/Plan-R1/base_weight/pred.ckpt'
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # 创建模型
    map_encoder = MapEncoderExportable()
    step_model = StepModel()

    # 复制权重
    copy_map_encoder_weights(checkpoint, map_encoder)
    copy_step_model_weights(checkpoint, step_model)

    map_encoder.eval().cuda()
    step_model.eval().cuda()
    print('✓ 模型加载完成')

    # 加载数据
    data_dir = '/workspace/planner/Plan-R1/cache_data'
    data_files = sorted(glob.glob(os.path.join(data_dir, '*.pt')))[:50]
    print(f'\n找到 {len(data_files)} 个数据文件')

    # 准备保存数据
    all_samples = {}
    num_samples = 0

    for i, data_file in enumerate(data_files):
        try:
            data = torch.load(data_file, map_location='cpu', weights_only=False)
            
            # 提取数据
            polygon = data['polygon'].unsqueeze(0).cuda()
            polyline = data['polyline'].unsqueeze(0).cuda()
            polygon_mask = data['polygon_mask'].unsqueeze(0).cuda()
            polyline_mask = data['polyline_mask'].unsqueeze(0).cuda()
            l2g_edge_index = data['l2g_edge_index'].cuda()
            l2g_edge_attr = data['l2g_edge_attr'].cuda()
            g2g_edge_index = data['g2g_edge_index'].cuda()
            g2g_edge_attr = data['g2g_edge_attr'].cuda()
            
            # MapEncoder
            with torch.no_grad():
                polygon_embs = map_encoder(
                    polygon, polyline, polygon_mask, polyline_mask,
                    l2g_edge_index, l2g_edge_attr, g2g_edge_index, g2g_edge_attr
                )
            
            # StepModel 输入
            num_agents = min(data['agent_type'].shape[0], 21)
            agent_token = data['recon_token'][:num_agents, :4].cuda()
            agent_type = data['agent_type'][:num_agents].cuda()
            agent_box = data['agent_box'][:num_agents].cuda()
            agent_identity = data['agent_identity'][:num_agents].cuda()
            
            # 填充到固定大小
            def pad_to_size(tensor, target_size, dim=0):
                if tensor.shape[dim] >= target_size:
                    return tensor.narrow(dim, 0, target_size)
                pad_shape = list(tensor.shape)
                pad_shape[dim] = target_size - tensor.shape[dim]
                padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
                return torch.cat([tensor, padding], dim=dim)
            
            agent_token = pad_to_size(agent_token, 21)
            agent_type = pad_to_size(agent_type, 21)
            agent_box = pad_to_size(agent_box, 21)
            agent_identity = pad_to_size(agent_identity, 21)
            
            # 构建边
            edges = build_step_edges(
                num_agents, agent_token.shape[1],
                polygon_embs.squeeze(0), agent_box,
                data['a2g_edge_index'].cuda() if 'a2g_edge_index' in data else None,
                data['a2g_edge_attr'].cuda() if 'a2g_edge_attr' in data else None
            )
            
            k2k_t_edge_index = edges['k2k_t_edge_index']
            k2k_t_edge_attr = edges['k2k_t_edge_attr']
            g2k_edge_index = edges['g2k_edge_index']
            g2k_edge_attr = edges['g2k_edge_attr']
            k2k_a_edge_index = edges['k2k_a_edge_index']
            k2k_a_edge_attr = edges['k2k_a_edge_attr']
            
            # StepModel 推理
            with torch.no_grad():
                output = step_model(
                    agent_token, agent_type, agent_box, agent_identity,
                    polygon_embs.squeeze(0),
                    k2k_t_edge_index, k2k_t_edge_attr,
                    g2k_edge_index, g2k_edge_attr,
                    k2k_a_edge_index, k2k_a_edge_attr
                )
            
            # 保存数据
            prefix = f'sample_{num_samples}'
            all_samples[f'{prefix}_output'] = output.cpu().numpy()
            all_samples[f'{prefix}_num_agents'] = num_agents
            all_samples[f'{prefix}_agent_token'] = agent_token.cpu().numpy()
            all_samples[f'{prefix}_agent_type'] = agent_type.cpu().numpy()
            all_samples[f'{prefix}_agent_box'] = agent_box.cpu().numpy()
            all_samples[f'{prefix}_agent_identity'] = agent_identity.cpu().numpy()
            all_samples[f'{prefix}_polygon_embs'] = polygon_embs.squeeze(0).cpu().numpy()
            all_samples[f'{prefix}_k2k_t_edge_index'] = k2k_t_edge_index.cpu().numpy()
            all_samples[f'{prefix}_k2k_t_edge_attr'] = k2k_t_edge_attr.cpu().numpy()
            all_samples[f'{prefix}_g2k_edge_index'] = g2k_edge_index.cpu().numpy()
            all_samples[f'{prefix}_g2k_edge_attr'] = g2k_edge_attr.cpu().numpy()
            all_samples[f'{prefix}_k2k_a_edge_index'] = k2k_a_edge_index.cpu().numpy()
            all_samples[f'{prefix}_k2k_a_edge_attr'] = k2k_a_edge_attr.cpu().numpy()
            
            num_samples += 1
            
            if (i + 1) % 10 == 0:
                print(f'  处理完成: {i + 1}/{len(data_files)}')
            
        except Exception as e:
            print(f'  跳过样本 {i}: {e}')
            import traceback
            traceback.print_exc()
            continue

    # 保存
    np.savez('/tmp/step_real_data_50.npz', **all_samples)
    print(f'\n✓ 保存到 /tmp/step_real_data_50.npz')
    print(f'✓ 共处理 {num_samples} 个样本')


if __name__ == "__main__":
    main()
