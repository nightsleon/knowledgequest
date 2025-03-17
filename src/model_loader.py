#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量模型加载器模块

@author liangjun
"""

import torch
import numpy as np

# 尝试导入 sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    print("警告: sentence-transformers 包未安装，无法使用文本向量化功能")
    print("请运行: pip install sentence-transformers 来安装它")

# 全局变量存储模型实例
_model = None

def load_model(model_name='moka-ai/m3e-base'):
    """
    加载 Sentence-Transformers 模型
    
    Args:
        model_name: 模型名称，默认使用专门为中文优化的开源模型
        
    Returns:
        加载的模型实例，如果加载失败则返回 None
    """
    global _model
    
    if not HAVE_SENTENCE_TRANSFORMERS:
        print("错误: 未安装 sentence-transformers 包，无法加载模型")
        return None
        
    if _model is None:
        try:
            print(f"正在加载 Sentence-Transformers 模型: {model_name}")
            # 首次使用时会下载模型，后续使用将从本地加载
            _model = SentenceTransformer(model_name)
            print("模型加载完成")
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            return None
    
    return _model

def text_to_vector(text):
    """
    将文本转换为向量
    
    使用 Sentence-Transformers 生成高质量的文本嵌入
    
    Args:
        text: 要嵌入的文本
        
    Returns:
        文本的向量表示，如果转换失败则返回 None
    """
    if not HAVE_SENTENCE_TRANSFORMERS:
        print("错误: 未安装 sentence-transformers 包，无法进行文本向量化")
        return None
        
    model = load_model()
    if model is None:
        print("错误: 模型加载失败，无法进行文本向量化")
        return None
        
    try:
        # 将文本转换为向量
        with torch.no_grad():
            embedding = model.encode(text, convert_to_numpy=True)
        # 归一化向量
        return embedding / np.linalg.norm(embedding)
    except Exception as e:
        print(f"生成文本嵌入时出错: {str(e)}")
        return None

def get_vector_dimension():
    """
    获取向量维度
    
    Returns:
        向量维度，如果模型未加载则返回默认值 768
    """
    model = load_model()
    if model is None:
        return 768  # 默认维度更新为 768，适应 moka-ai/m3e-base 模型
    
    try:
        # 获取模型输出维度
        return model.get_sentence_embedding_dimension()
    except Exception as e:
        print(f"获取向量维度时出错: {str(e)}")
        return 768  # 默认维度更新为 768，适应 moka-ai/m3e-base 模型
