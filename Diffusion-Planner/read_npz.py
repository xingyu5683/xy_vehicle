#!/usr/bin/env python3
"""
读取并显示 npz 文件内容
用法: 
    python read_npz.py [file_path]
    如果不提供路径，会自动查找最新的 npz 文件
"""
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime

def find_latest_npz(directory):
    """查找目录中最新的 npz 文件"""
    npz_files = list(Path(directory).glob("*.npz"))
    if not npz_files:
        return None
    
    # 按修改时间排序
    npz_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return npz_files[0]

def read_and_display_npz(file_path):
    """读取并显示 npz 文件内容"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return
    
    print("=" * 80)
    print(f"📁 文件路径: {file_path}")
    print(f"📊 文件大小: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"🕒 修改时间: {datetime.fromtimestamp(file_path.stat().st_mtime)}")
    print("=" * 80)
    
    try:
        data = np.load(file_path)
        print(f"\n✅ 成功加载文件")
        print(f"📋 包含的键 (keys): {len(data.keys())} 个\n")
        
        # 显示所有键和数据信息
        for i, key in enumerate(data.keys(), 1):
            value = data[key]
            print(f"{i}. 键名: {key}")
            print(f"   形状 (shape): {value.shape}")
            print(f"   数据类型 (dtype): {value.dtype}")
            print(f"   大小: {value.nbytes / 1024 / 1024:.2f} MB")
            
            # 显示数据统计信息（如果是数值类型）
            if np.issubdtype(value.dtype, np.number):
                print(f"   最小值: {np.min(value):.6f}")
                print(f"   最大值: {np.max(value):.6f}")
                print(f"   平均值: {np.mean(value):.6f}")
                if value.size > 0:
                    print(f"   非零元素: {np.count_nonzero(value)} / {value.size}")
            
            # 显示前几个元素（如果是一维或可以展平）
            if value.size <= 20:
                print(f"   数据内容: {value}")
            elif value.ndim == 1:
                print(f"   前5个元素: {value[:5]}")
                print(f"   后5个元素: {value[-5:]}")
            elif value.ndim == 2:
                print(f"   前3x3数据:\n{value[:3, :3]}")
            elif value.ndim == 3:
                print(f"   形状示例: [0, :3, :3]\n{value[0, :3, :3]}")
            
            print()
        
        data.close()
        print("=" * 80)
        print("✅ 读取完成")
        
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    if len(sys.argv) >= 2:
        # 使用命令行参数指定的文件
        file_path = sys.argv[1]
    else:
        # 自动查找最新的 npz 文件
        # 默认查找目录
        default_dirs = [
            "/mnt/data/output/preprocess_data",
            "./cache",
            "/tmp/preprocess_data",
            os.getcwd()
        ]
        
        file_path = None
        for directory in default_dirs:
            if os.path.exists(directory):
                file_path = find_latest_npz(directory)
                if file_path:
                    print(f"🔍 在 {directory} 中找到最新的 npz 文件")
                    break
        
        if not file_path:
            print("❌ 未找到 npz 文件")
            print(f"   请指定文件路径: python {sys.argv[0]} <file_path>")
            print(f"   或在以下目录中放置 npz 文件: {', '.join(default_dirs)}")
            sys.exit(1)
    
    read_and_display_npz(file_path)

if __name__ == "__main__":
    main()

