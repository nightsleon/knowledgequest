#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大模型交互命令行工具

@author liangjun
"""

import sys
import argparse
from src.app import VectorApp

def main():
    """
    主函数，处理命令行参数并执行相应操作
    """
    parser = argparse.ArgumentParser(description="与大模型交互，基于向量检索回答问题")
    parser.add_argument("query", nargs="*", help="查询问题")
    parser.add_argument("-l", "--limit", type=int, default=3, help="检索结果数量上限，默认为3")
    
    args = parser.parse_args()
    
    # 初始化应用
    print("正在初始化应用...")
    app = VectorApp()
    
    # 如果没有提供查询，进入交互模式
    if not args.query:
        print("\n欢迎使用大模型交互系统！输入 'exit' 或 'quit' 退出。")
        while True:
            try:
                query = input("\n请输入您的问题: ")
                if query.lower() in ["exit", "quit", "退出"]:
                    break
                    
                if not query.strip():
                    continue
                    
                # 调用大模型回答问题
                response = app.chat(query, limit=args.limit)
                print(f"\n回答: {response}")
                
            except KeyboardInterrupt:
                print("\n程序已中断")
                break
            except Exception as e:
                print(f"\n发生错误: {str(e)}")
    else:
        # 使用命令行参数中的查询
        query = " ".join(args.query)
        print(f"查询: {query}")  # 打印查询内容以确认正确性
        response = app.chat(query, limit=args.limit)
        print(f"\n回答: {response}")
    
    # 关闭应用
    app.close()
    print("应用已关闭")

if __name__ == "__main__":
    main()
