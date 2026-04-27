#!/usr/bin/env python3
"""
从指定目录读取文件名并保存到 JSON 文件
用法: python list_files_to_json.py [目录路径] [输出JSON路径] [文件数量]
"""

import os
import json
import sys
from pathlib import Path


def list_files_to_json(directory, output_json, num_files=2000):
    """
    从目录中读取指定数量的文件名并保存到 JSON 文件
    
    Args:
        directory: 要读取的目录路径
        output_json: 输出的 JSON 文件路径
        num_files: 要读取的文件数量，默认 100
    """
    # 检查目录是否存在
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    if not dir_path.is_dir():
        raise NotADirectoryError(f"路径不是目录: {directory}")
    
    # 获取所有文件（不包括子目录）
    all_files = [f.name for f in dir_path.iterdir() if f.is_file()]
    
    # 排序以确保结果一致
    all_files.sort()
    
    # 限制文件数量
    selected_files = all_files[:num_files]
    
    # 准备输出数据
    output_data = {
        "directory": str(dir_path.absolute()),
        "total_files_in_directory": len(all_files),
        "selected_files_count": len(selected_files),
        "files": selected_files
    }
    
    # 保存到 JSON 文件
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 成功读取 {len(selected_files)} 个文件名")
    print(f"📁 目录: {directory}")
    print(f"📄 总文件数: {len(all_files)}")
    print(f"💾 已保存到: {output_json}")
    
    return output_data


def main():
    """主函数"""
    # 默认参数
    default_directory = "/mnt/data/output/preprocess_data"
    default_output = "file_list.json"
    default_num_files = 3000
    
    # 解析命令行参数
    if len(sys.argv) >= 2:
        directory = sys.argv[1]
    else:
        directory = default_directory
    
    if len(sys.argv) >= 3:
        output_json = sys.argv[2]
    else:
        output_json = default_output
    
    if len(sys.argv) >= 4:
        try:
            num_files = int(sys.argv[3])
        except ValueError:
            print(f"⚠️  无效的文件数量参数: {sys.argv[3]}，使用默认值 {default_num_files}")
            num_files = default_num_files
    else:
        num_files = default_num_files
    
    try:
        list_files_to_json(directory, output_json, num_files)
    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

