#!/usr/bin/env python3
"""
将 nuplan_train_test.json 中对应的数据库文件移动到 /mnt/datadownload 文件夹
"""

import json
import os
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm


def find_db_file(db_name: str, search_dirs: list) -> str:
    """
    在指定目录中查找数据库文件
    
    Args:
        db_name: 数据库文件名（不含扩展名），例如 "2021.09.06.05.56.29_veh-51_00825_00944"
        search_dirs: 搜索目录列表
    
    Returns:
        找到的文件路径，如果未找到返回 None
    """
    db_filename = f"{db_name}.db"
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        
        # 先检查常见子目录（trainval, mini, test等），提高查找速度
        common_subdirs = ['trainval', 'mini', 'test', 'val']
        for subdir in common_subdirs:
            subdir_path = os.path.join(search_dir, subdir)
            if os.path.exists(subdir_path):
                file_path = os.path.join(subdir_path, db_filename)
                if os.path.exists(file_path):
                    return file_path
        
        # 如果常见目录中没找到，递归搜索
        for root, dirs, files in os.walk(search_dir):
            if db_filename in files:
                return os.path.join(root, db_filename)
    
    return None


def main():
    parser = argparse.ArgumentParser(description='将 nuplan_train_test.json 中的文件移动到 /mnt/datadownload')
    parser.add_argument('--json_file', type=str, 
                       default='nuplan_train_test.json',
                       help='包含文件名的 JSON 文件路径')
    parser.add_argument('--source_dirs', type=str, nargs='+',
                       default=['/mnt/data/dataset/nuplan-v1.1/splits'],
                       help='源文件搜索目录列表（会递归搜索）')
    parser.add_argument('--target_dir', type=str,
                       default='/mnt/datadownload',
                       help='目标目录路径')
    parser.add_argument('--dry_run', action='store_true',
                       help='仅显示将要移动的文件，不实际移动')
    
    args = parser.parse_args()
    
    # 读取 JSON 文件
    json_path = args.json_file
    if not os.path.isabs(json_path):
        json_path = os.path.join(os.path.dirname(__file__), json_path)
    
    print(f"读取 JSON 文件: {json_path}", flush=True)
    with open(json_path, 'r') as f:
        db_names = json.load(f)
    
    print(f"找到 {len(db_names)} 个数据库文件名", flush=True)
    
    # 创建目标目录
    target_dir = Path(args.target_dir)
    if not args.dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"目标目录: {target_dir}", flush=True)
    else:
        print(f"目标目录（模拟）: {target_dir}", flush=True)
    
    # 统计信息
    found_files = []
    not_found_files = []
    moved_files = []
    error_files = []
    
    print(f"\n开始查找和移动文件...", flush=True)
    print(f"搜索目录: {args.source_dirs}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # 遍历所有数据库文件名
    for db_name in tqdm(db_names, desc="处理文件"):
        # 查找文件
        source_file = find_db_file(db_name, args.source_dirs)
        
        if source_file is None:
            not_found_files.append(db_name)
            continue
        
        found_files.append(db_name)
        target_file = target_dir / f"{db_name}.db"
        
        if args.dry_run:
            print(f"  [模拟] {source_file} -> {target_file}", flush=True)
            moved_files.append(db_name)
        else:
            try:
                # 如果目标文件已存在，跳过或覆盖（根据需求修改）
                if target_file.exists():
                    print(f"  警告: 目标文件已存在，跳过: {target_file}", flush=True)
                    continue
                
                # 移动文件
                shutil.move(source_file, target_file)
                moved_files.append(db_name)
            except Exception as e:
                print(f"  错误: 移动文件失败 {source_file}: {e}", flush=True)
                error_files.append((db_name, str(e)))
    
    # 打印统计信息
    print(f"\n{'='*60}", flush=True)
    print(f"处理完成！", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"统计信息:", flush=True)
    print(f"  总文件数: {len(db_names)}", flush=True)
    print(f"  找到文件: {len(found_files)}", flush=True)
    print(f"  未找到文件: {len(not_found_files)}", flush=True)
    print(f"  成功移动: {len(moved_files)}", flush=True)
    print(f"  移动失败: {len(error_files)}", flush=True)
    
    if not_found_files:
        print(f"\n未找到的文件（前10个）:", flush=True)
        for db_name in not_found_files[:10]:
            print(f"  {db_name}", flush=True)
        if len(not_found_files) > 10:
            print(f"  ... 还有 {len(not_found_files) - 10} 个文件未找到", flush=True)
    
    if error_files:
        print(f"\n移动失败的文件:", flush=True)
        for db_name, error in error_files[:10]:
            print(f"  {db_name}: {error}", flush=True)
        if len(error_files) > 10:
            print(f"  ... 还有 {len(error_files) - 10} 个文件移动失败", flush=True)
    
    if not args.dry_run:
        print(f"\n✓ 文件已移动到: {target_dir}", flush=True)
    else:
        print(f"\n⚠ 这是模拟运行，文件未被实际移动", flush=True)
        print(f"   移除 --dry_run 参数以实际移动文件", flush=True)


if __name__ == '__main__':
    main()

