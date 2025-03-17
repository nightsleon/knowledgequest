#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Milvus 向量数据库示例程序 - 使用 Sentence-Transformers 进行文本嵌入

@author liangjun
"""

from pymilvus import MilvusClient
import numpy as np
import os
import torch

# 尝试导入 sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    print("警告: sentence-transformers 包未安装，将使用备用的关键词匹配方法")
    print("请运行: pip install sentence-transformers 来安装它")

# 全局变量存储模型实例
model = None

def load_sentence_transformer_model(model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    """
    加载 Sentence-Transformers 模型
    
    Args:
        model_name: 模型名称，默认使用支持多语言（包括中文）的轻量级模型
        
    Returns:
        加载的模型实例
    """
    global model
    if not HAVE_SENTENCE_TRANSFORMERS:
        return None
        
    if model is None:
        try:
            print(f"正在加载 Sentence-Transformers 模型: {model_name}")
            # 首次使用时会下载模型，后续使用将从本地加载
            model = SentenceTransformer(model_name)
            print("模型加载完成")
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            return None
    
    return model

# 使用 Sentence-Transformers 或备用方法将文本转换为向量
def text_to_vector(text, dimension=384):
    """
    将文本转换为向量
    
    如果安装了 sentence-transformers，将使用它生成高质量的文本嵌入
    否则将回退到基于关键词的匹配方法
    
    Args:
        text: 要嵌入的文本
        dimension: 向量维度（仅在使用备用方法时适用）
        
    Returns:
        文本的向量表示
    """
    # 尝试使用 Sentence-Transformers
    if HAVE_SENTENCE_TRANSFORMERS:
        transformer_model = load_sentence_transformer_model()
        if transformer_model is not None:
            # 使用模型生成嵌入
            try:
                # 将文本转换为向量
                with torch.no_grad():
                    embedding = transformer_model.encode(text, convert_to_numpy=True)
                # 归一化向量
                return embedding / np.linalg.norm(embedding)
            except Exception as e:
                print(f"生成嵌入时出错: {str(e)}")
                # 如果出错，回退到备用方法
    
    # 备用方法: 使用关键词匹配
    print("使用备用的关键词匹配方法生成向量")
    
    # 定义一些关键词类别
    categories = {
        '苹果': ['苹果', 'iPhone', '手机', '红富士', '新疆'],
        '拍照': ['拍照', '相机', '照片', '拍摄'],
        '性能': ['性能', '强劲', '强大', '表现'],
        '人工智能': ['人工智能', 'AI', '智能', '深度学习'],
        '医疗': ['医疗', '诊断', '疾病', '医院', '医生']
    }
    
    # 初始化向量
    vector = np.zeros(dimension)
    
    # 分配各类别的特征空间
    segment_size = dimension // len(categories)
    
    # 根据文本中出现的关键词设置向量值
    for i, (category, keywords) in enumerate(categories.items()):
        # 计算该类别的权重
        weight = 0
        for keyword in keywords:
            if keyword in text:
                weight += 1
        
        # 如果有关键词匹配，设置该类别的向量值
        if weight > 0:
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            # 使用文本的哈希作为种子，但主要由关键词匹配决定
            np.random.seed(sum(ord(c) for c in category + text))
            vector[start_idx:end_idx] = np.random.uniform(0.5, 1.0, segment_size) * weight
    
    # 如果向量全为零（没有关键词匹配），生成一个随机向量
    if np.all(vector == 0):
        np.random.seed(sum(ord(c) for c in text))
        vector = np.random.uniform(-0.1, 0.1, dimension)
    
    # 归一化向量，使其适合内积度量
    return vector / np.linalg.norm(vector)

# 初始化 Milvus 客户端并创建集合
def init_milvus():
    """初始化 Milvus 客户端并创建集合"""
    # 确保数据目录存在
    os.makedirs("./data", exist_ok=True)
    
    # 创建 Milvus 客户端
    client = MilvusClient("./data/milvus_demo.db")
    
    # 如果集合已存在，则删除它
    if "demo_collection" in client.list_collections():
        client.drop_collection("demo_collection")
    
    # 创建新集合，使用内积(IP)作为度量类型
    client.create_collection(
        collection_name="demo_collection",
        dimension=384,
        metric_type="IP"  # 使用内积作为相似度度量
    )
    
    return client

# 插入示例数据
def insert_sample_data(client):
    """插入示例数据到 Milvus"""
    docs = [
        "苹果 iPhone 15 具有出色的拍照功能和强大的性能。",
        "iPhone 15 这款苹果手机拍照效果很棒，性能也十分强劲。",
        "苹果新出的 iPhone 15 在拍照和性能方面表现优异。",
        "烟台红富士苹果色泽口感都不错。",
        "新疆苹果表现也很优异。",
        "人工智能技术正在深刻改变医疗行业的诊断方式。",
        "AI 技术对医疗行业的诊断方法带来了重大变革。",
        "人工智能正在极大地影响着医疗领域的疾病诊断手段。",
    ]
    
    # 将文本转换为向量
    vectors = [text_to_vector(doc) for doc in docs]
    
    # 准备插入数据
    data = [{
        "id": i, 
        "vector": vectors[i], 
        "text": docs[i], 
        "subject": "title"
    } for i in range(len(vectors))]
    
    # 插入数据
    res = client.insert(
        collection_name="demo_collection",
        data=data
    )
    print(f"插入数据结果: {res}")
    return docs

# 按文本搜索
def search_by_text(client, query_text, limit=3):
    """
    按文本搜索相似内容
    
    Args:
        client: Milvus 客户端
        query_text: 查询文本
        limit: 返回结果数量上限
        
    Returns:
        搜索结果列表
    """
    # 将查询文本转换为向量
    query_vector = text_to_vector(query_text)
    
    # 执行向量搜索
    results = client.search(
        collection_name="demo_collection",
        data=[query_vector],
        filter="subject == 'title'",
        limit=limit,
        output_fields=["text", "subject"],
    )
    
    return results

# 查询所有数据
def query_all_data(client):
    """查询所有数据"""
    results = client.query(
        collection_name="demo_collection",
        filter="subject == 'title'",
        output_fields=["text", "subject"],
    )
    return results

# 删除所有数据
def delete_all_data(client):
    """删除所有数据"""
    results = client.delete(
        collection_name="demo_collection",
        filter="subject == 'title'",
    )
    return results

# 主函数
def main():
    # 如果安装了 sentence-transformers，预先加载模型
    if HAVE_SENTENCE_TRANSFORMERS:
        load_sentence_transformer_model()
    
    # 初始化 Milvus
    client = init_milvus()
    print("Milvus 初始化完成")
    
    # 插入示例数据
    docs = insert_sample_data(client)
    print(f"插入了 {len(docs)} 条示例数据")
    
    # 按文本搜索 - 苹果手机相关
    query_text = "苹果手机拍照怎么样？"
    results = search_by_text(client, query_text, limit=2)
    print(f"查询: '{query_text}'")
    for i, item in enumerate(results[0]):
        print(f"  {i+1}. 相似度: {item['distance']:.4f}, 文本: {item['entity']['text']}")
    print()
    
    # 按文本搜索 - 人工智能相关
    query_text = "人工智能对医疗领域影响怎么样？"
    results = search_by_text(client, query_text, limit=2)
    print(f"查询: '{query_text}'")
    for i, item in enumerate(results[0]):
        print(f"  {i+1}. 相似度: {item['distance']:.4f}, 文本: {item['entity']['text']}")
    print()
    
    # 按文本搜索 - 自定义查询
    custom_query = input("请输入您的查询文本 (或直接回车跳过): ")
    if custom_query.strip():
        results = search_by_text(client, custom_query, limit=3)
        print(f"查询: '{custom_query}'")
        for i, item in enumerate(results[0]):
            print(f"  {i+1}. 相似度: {item['distance']:.4f}, 文本: {item['entity']['text']}")
        print()
    
    # 查询所有数据
    all_data = query_all_data(client)
    print("所有数据:")
    for i, item in enumerate(all_data):
        print(f"  {i+1}. {item}")
    print()
    
    # 删除所有数据
    delete_result = delete_all_data(client)
    print(f"删除结果: {delete_result}")
    
    print("演示完成")

# 入口点，使 PyCharm 可以直接运行此文件
if __name__ == "__main__":
    main()
