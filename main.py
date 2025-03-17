#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量数据库应用主程序

@author liangjun
"""

import os
import sys
import argparse

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.cli import CLI

def parse_args():
    """
    解析命令行参数
    
    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description="向量数据库交互式应用")
    parser.add_argument(
        "--db-path", 
        type=str, 
        default="./data/milvus_db.db",
        help="数据库存储路径，必须以 .db 结尾"
    )
    parser.add_argument(
        "--collection", 
        type=str, 
        default="text_collection",
        help="集合名称"
    )
    
    return parser.parse_args()

def main():
    """
    主程序入口
    """
    args = parse_args()
    
    # 创建数据库目录
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
    
    # 初始化并运行 CLI
    cli = CLI(args.db_path, args.collection)
    cli.run()

if __name__ == "__main__":
    main()
