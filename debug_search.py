#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试 Milvus 搜索功能

@author liangjun
"""

from src.app import VectorApp

def main():
    print("正在初始化应用...")
    app = VectorApp()
    
    # 查询所有数据
    print("\n查询所有数据:")
    all_data = app.db.query(output_fields=["id", "text", "subject"])
    print(f"数据条数: {len(all_data) if all_data else 0}")
    if all_data and len(all_data) > 0:
        print(f"第一条数据: {all_data[0]}")
        print(f"数据字段: {all_data[0].keys() if isinstance(all_data[0], dict) else 'Not a dict'}")
    
    # 测试搜索功能
    print("\n测试搜索功能:")
    results = app.search_text("人工智能")
    print(f"搜索结果: {results}")
    
    # 检查 Milvus 原始搜索结果
    print("\n检查 Milvus 原始搜索结果:")
    from src.model_loader import text_to_vector
    query_vector = text_to_vector("人工智能")
    if query_vector is not None:
        raw_results = app.db.client.search(
            collection_name=app.db.collection_name,
            data=[query_vector],
            filter="",
            limit=5,
            output_fields=["text", "subject"],
            search_params={
                "metric_type": "IP",
                "params": {"nprobe": 10}
            }
        )
        print(f"原始结果类型: {type(raw_results)}")
        print(f"原始结果: {raw_results}")
        if raw_results and len(raw_results) > 0 and len(raw_results[0]) > 0:
            print(f"第一个结果详情: {raw_results[0][0]}")
            print(f"结果类型: {type(raw_results[0][0])}")
            print(f"结果键值: {list(raw_results[0][0].keys()) if isinstance(raw_results[0][0], dict) else 'Not a dict'}")

if __name__ == "__main__":
    main()
