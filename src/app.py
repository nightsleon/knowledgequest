#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交互式向量数据库应用

@author liangjun
"""

import os
import sys
import json
from typing import List, Dict, Any, Tuple

from .model_loader import text_to_vector, load_model
from .vector_db import VectorDB
from .text_splitter import MarkdownSplitter

class VectorApp:
    """
    交互式向量数据库应用
    
    提供交互式的插入文本、删除文本向量、搜索文本等功能
    """
    
    def __init__(self, db_path="./data/milvus_db.db", collection_name="text_collection"):
        """
        初始化应用
        
        Args:
            db_path: 数据库存储路径，必须以 .db 结尾
            collection_name: 集合名称
        """
        # 初始化向量数据库
        self.db = VectorDB(db_path, collection_name)
        
        # 初始化模型
        load_model()
        
        # 初始化数据库
        self._init_database()
        
        # 存储已插入数据的ID和文本映射，方便管理
        self.id_text_map = {}
        self._load_existing_data()
    
    def _init_database(self):
        """
        初始化数据库
        """
        if not self.db.connect():
            print("错误: 无法连接到数据库")
            sys.exit(1)
            
        if not self.db.create_collection(drop_exists=False):
            print("错误: 无法创建集合")
            sys.exit(1)
    
    def _load_existing_data(self):
        """
        加载现有数据到内存映射
        """
        results = self.db.query(output_fields=["id", "text"])
        if results:
            for item in results:
                if "id" in item and "text" in item:
                    self.id_text_map[item["id"]] = item["text"]
            
            print(f"已加载 {len(self.id_text_map)} 条现有数据")
    
    def insert_text(self, text, subject="general", metadata=None):
        """
        插入文本
        
        Args:
            text: 要插入的文本
            subject: 文本主题，默认为 "general"
            metadata: 元数据字典，默认为 None
            
        Returns:
            成功返回插入的 ID，失败返回 None
        """
        # @author liangjun
        # 将文本转换为向量
        vector = text_to_vector(text)
        if vector is None:
            print("错误: 无法将文本转换为向量")
            return None
        
        # 插入数据
        doc_id = self.db.insert(text, vector, subject, metadata)
        if doc_id is not None:
            # 更新内存映射
            self.id_text_map[doc_id] = text
            print(f"文本插入成功，ID: {doc_id}, 主题: {subject}")
        
        return doc_id
    
    def insert_batch_texts(self, texts, subjects=None, metadata_list=None):
        """
        批量插入文本
        
        Args:
            texts: 要插入的文本列表
            subjects: 文本主题列表，默认为 None，如果为 None 则所有条目使用 "general" 主题
            metadata_list: 元数据列表，默认为 None
            
        Returns:
            成功返回插入的 ID 列表，失败返回 None
        """
        # @author liangjun
        
        if not texts:
            print("错误: 文本列表不能为空")
            return None
        
        # 将文本转换为向量
        vectors = []
        valid_texts = []
        valid_subjects = []
        valid_metadata = []
        
        for i, text in enumerate(texts):
            vector = text_to_vector(text)
            if vector is None:
                print(f"警告: 第 {i+1} 条文本无法转换为向量，已跳过")
                continue
                
            vectors.append(vector)
            valid_texts.append(text)
            
            # 处理主题
            if subjects is not None and i < len(subjects):
                valid_subjects.append(subjects[i])
            
            # 处理元数据
            if metadata_list is not None and i < len(metadata_list):
                valid_metadata.append(metadata_list[i])
            else:
                valid_metadata.append(None)
        
        if not vectors:
            print("错误: 所有文本都无法转换为向量")
            return None
        
        # 批量插入数据
        doc_ids = self.db.insert_batch(valid_texts, vectors, valid_subjects, valid_metadata)
        if doc_ids is not None:
            # 更新内存映射
            for i, doc_id in enumerate(doc_ids):
                self.id_text_map[doc_id] = valid_texts[i]
            
            print(f"批量插入成功，共 {len(doc_ids)} 条数据")
        
        return doc_ids
    
    def delete_by_id(self, doc_id):
        """
        根据 ID 删除文本
        
        Args:
            doc_id: 要删除的文本 ID
            
        Returns:
            成功返回 True，失败返回 False
        """
        if doc_id not in self.id_text_map:
            print(f"错误: ID {doc_id} 不存在")
            return False
        
        # 删除数据
        result = self.db.delete_by_ids(doc_id)
        if result and len(result) > 0:
            # 更新内存映射
            text = self.id_text_map.pop(doc_id, None)
            print(f"文本删除成功，ID: {doc_id}, 文本: {text}")
            return True
        
        return False
    
    def search_text(self, query_text, limit=5, include_subdirectories=True):
        """
        搜索相似文本
        
        Args:
            query_text: 查询文本
            limit: 返回结果数量上限，默认为 3
            include_subdirectories: 是否包含子目录相关内容，默认为 True
            
        Returns:
            搜索结果列表，失败返回 None
        """
        # @author liangjun
        # 将查询文本转换为向量
        query_vector = text_to_vector(query_text)
        if query_vector is None:
            print("错误: 无法将查询文本转换为向量")
            return None
        
        # 执行向量搜索，包含更多元数据字段以便展示完整上下文
        output_fields = ["text", "subject", "metadata", "title_path", "full_context", "parent_summary"]
        results = self.db.search(
            query_vector, 
            limit=limit, 
            output_fields=output_fields,
            include_subdirectories=include_subdirectories
        )
        
        if results is None or len(results) == 0:
            print("未找到相关文本")
            return None
        
        # 处理搜索结果，直接返回已格式化的结果
        # @author liangjun
        print(f"搜索结果类型: {type(results)}")
        if not results:
            print("搜索结果为 None")
            return None
        elif len(results) == 0:
            print("搜索结果为空列表")
            return None
        elif len(results[0]) == 0:
            print("搜索结果的第一项为空列表")
            return None
            
        # 直接使用 vector_db.py 中已经格式化的结果
        # 确保所有必要字段都存在
        print(f"开始格式化结果，结果数量: {len(results[0])}")
        formatted_results = []
        for hit in results[0]:
            # 打印每个结果项的字段
            print(f"结果项字段: {hit.keys()}")
            
            # 确保基本字段存在
            result_item = {
                "id": hit.get("id"),
                "score": hit.get("score"),
                "text": hit.get("text", ""),
                "subject": hit.get("subject", "general")
            }
            
            # 安全地复制其他字段
            for key, value in hit.items():
                if key not in ["id", "score", "text", "subject"] and value is not None:
                    result_item[key] = value
                    
            # 确保上下文字段的一致性
            # 将 full_context 映射到 context
            if "full_context" in hit and hit["full_context"] is not None:
                result_item["context"] = hit["full_context"]
                
            # 确保子目录相关字段的一致性
            if "is_subdirectory" in hit and hit["is_subdirectory"]:
                result_item["is_subdirectory"] = True
                result_item["level"] = hit.get("level", 0)
            
            formatted_results.append(result_item)
        
        print(f"格式化完成，返回 {len(formatted_results)} 条结果")
        return formatted_results
    
    def list_all_texts(self, subject=None):
        """
        列出所有文本
        
        Args:
            subject: 按主题筛选，默认为 None 表示不筛选
            
        Returns:
            文本列表，失败返回空列表
        """
        # @author liangjun
        # 构建过滤表达式
        filter_expr = None
        if subject:
            filter_expr = f"subject == '{subject}'"
            
        # 查询数据库
        results = self.db.query(filter_expr=filter_expr, output_fields=["id", "text", "subject"])
        if not results:
            return []
        
        return [(item["id"], item["text"], item.get("subject", "general")) for item in results if "id" in item and "text" in item]
    
    def count(self):
        """
        获取数据数量
        
        Returns:
            数据数量
        """
        return self.db.count()
    
    def clear_all(self, subject=None):
        """
        清除数据
        
        Args:
            subject: 要清除的数据主题，如果为 None 则清除所有数据
        
        Returns:
            成功返回 True，失败返回 False
        """
        # @author liangjun
        result = self.db.clear_collection(subject)
        
        if result and subject is None:
            # 如果清除所有数据，则清空内存映射
            self.id_text_map.clear()
        elif result and subject is not None:
            # 如果清除特定主题的数据，需要重新加载数据
            # 查询剩余的数据
            self.id_text_map.clear()  # 先清空内存映射
            
            # 重新加载数据
            results = self.db.query(output_fields=["id", "text"])
            if results:
                for item in results:
                    if "id" in item and "text" in item:
                        self.id_text_map[item["id"]] = item["text"]
                        
        return result
    
    def split_markdown_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """
        切分 Markdown 文件并返回切片结果
        
        Args:
            file_path: Markdown 文件路径
            chunk_size: 每个切片的最大字符数
            chunk_overlap: 切片之间的重叠字符数
            
        Returns:
            切分后的文本块列表，每个块包含文本内容和元数据
        """
        # @author liangjun
        splitter = MarkdownSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_file(file_path)
    
    def insert_markdown_chunks(self, chunks: List[Dict[str, Any]], subject: str = "markdown") -> Tuple[int, List[int]]:
        """
        插入 Markdown 切片到数据库
        
        Args:
            chunks: 切片列表，每个切片包含文本内容和元数据
            subject: 文本主题
            
        Returns:
            成功插入的数量和插入的 ID 列表
        """
        # @author liangjun
        if not chunks:
            return 0, []
        
        # 准备批量插入的数据
        texts = []
        metadatas = []
        
        for chunk in chunks:
            # 提取文本和元数据
            text = chunk["text"]
            metadata = chunk["metadata"]
            
            # 将元数据转换为 JSON 字符串并存储
            metadata_str = json.dumps(metadata, ensure_ascii=False)
            
            texts.append(text)
            metadatas.append({"metadata": metadata_str})
        
        # 批量插入到数据库
        subjects = [subject] * len(texts)  # 创建与文本数量相同的主题列表
        return len(texts), self.insert_batch_texts(texts, subjects, metadatas)
    
    def process_markdown_file(self, file_path: str, subject: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> Tuple[int, List[int]]:
        """
        处理 Markdown 文件，切分并插入到数据库
        
        Args:
            file_path: Markdown 文件路径
            subject: 文本主题，如果为 None 则使用文件名
            chunk_size: 每个切片的最大字符数
            chunk_overlap: 切片之间的重叠字符数
            
        Returns:
            成功插入的数量和插入的 ID 列表
        """
        # @author liangjun
        # 如果没有指定主题，使用文件名作为主题
        if subject is None:
            subject = os.path.splitext(os.path.basename(file_path))[0]
        
        # 切分 Markdown 文件
        chunks = self.split_markdown_file(file_path, chunk_size, chunk_overlap)
        
        if not chunks:
            return 0, []
        
        # 插入切片到数据库
        return self.insert_markdown_chunks(chunks, subject)
        
    def chat(self, query_text, limit=3):
        """
        与大模型交互，基于检索结果回答问题
        
        Args:
            query_text: 用户问题
            limit: 返回结果数量上限，默认为 3
            
        Returns:
            大模型回答，失败返回 None
        """
        # @author liangjun
        # 导入 LLMChat 类
        from .llm_chat import LLMChat
        
        # 首先使用向量检索获取相关文档
        print(f"正在检索与问题相关的文档: {query_text}")
        search_results = self.search_text(query_text, limit=limit)
        
        if not search_results:
            print("未找到相关文档，无法提供上下文")
            return "抱歉，我无法找到与您问题相关的信息。"
            
        print(f"找到 {len(search_results)} 条相关文档")
        
        # 初始化大模型交互类
        llm_chat = LLMChat(model_name="qwen2.5:1.5b")
        
        # 使用大模型回答问题
        response = llm_chat.chat_with_context(query_text, search_results)
        
        if response is None:
            return "抱歉，大模型处理您的问题时出现错误。"
            
        return response
    
    def close(self):
        """
        关闭应用
        """
        self.db.close()
