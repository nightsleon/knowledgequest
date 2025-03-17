#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Milvus 向量数据库操作模块

@author liangjun
"""

import os
from pymilvus import MilvusClient
from .model_loader import get_vector_dimension

class VectorDB:
    """
    向量数据库管理类
    
    封装了 Milvus 向量数据库的常用操作，包括创建集合、插入数据、搜索、查询和删除等功能
    """
    
    def __init__(self, db_path="./data/milvus_db.db", collection_name="text_collection"):
        """
        初始化向量数据库
        
        Args:
            db_path: 数据库存储路径，必须以 .db 结尾
            collection_name: 集合名称
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.dimension = get_vector_dimension()
        self.last_id = 0  # 用于生成唯一ID
        
        # 创建数据库目录
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    def connect(self):
        """
        连接到 Milvus 数据库
        
        Returns:
            成功返回 True，失败返回 False
        """
        try:
            # 直接使用本地数据库文件路径
            self.client = MilvusClient(self.db_path)
            return True
        except Exception as e:
            print(f"连接到 Milvus 数据库失败: {str(e)}")
            return False
            
    def collection_exists(self):
        """
        检查集合是否存在
        
        Returns:
            如果集合存在返回 True，否则返回 False
        """
        if self.client is None and not self.connect():
            return False
            
        try:
            collections = self.client.list_collections()
            return self.collection_name in collections
        except Exception as e:
            print(f"检查集合存在失败: {str(e)}")
            return False
    
    def create_collection(self, drop_exists=False):
        """
        创建集合
        
        Args:
            drop_exists: 如果集合已存在，是否删除它
            
        Returns:
            成功返回 True，失败返回 False
        """
        if self.client is None and not self.connect():
            return False
            
        try:
            # 检查集合是否存在
            if self.collection_name in self.client.list_collections():
                if drop_exists:
                    self.client.drop_collection(self.collection_name)
                    print(f"集合 {self.collection_name} 已存在，已删除")
                else:
                    print(f"集合 {self.collection_name} 已存在，将继续使用")
                    return True
            
            # 创建新集合，使用内积(IP)作为度量类型
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                metric_type="IP"  # 使用内积作为相似度度量
            )
            print(f"集合 {self.collection_name} 创建成功")
            return True
        except Exception as e:
            print(f"创建集合失败: {str(e)}")
            return False
    
    def insert(self, text, vector, subject="general", metadata=None):
        """
        插入单条数据
        
        Args:
            text: 文本内容
            vector: 文本对应的向量
            subject: 主题分类，默认为 "general"
            metadata: 元数据字典，默认为 None
            
        Returns:
            成功返回插入的 ID，失败返回 None
        """
        if self.client is None and not self.connect():
            return None
            
        try:
            # 准备插入数据
            data = {
                "id": self._generate_id(),  # 生成唯一ID
                "vector": vector,
                "text": text,
                "subject": subject  # 添加主题字段
            }
            
            # 添加元数据
            if metadata is not None and isinstance(metadata, dict):
                data.update(metadata)
            
            # 插入数据
            result = self.client.insert(
                collection_name=self.collection_name,
                data=[data]
            )
            
            if result and 'ids' in result and len(result['ids']) > 0:
                return result['ids'][0]
            return None
        except Exception as e:
            print(f"插入数据失败: {str(e)}")
            return None
    
    def insert_batch(self, texts, vectors, subjects=None, metadata_list=None):
        """
        批量插入数据
        
        Args:
            texts: 文本列表
            vectors: 向量列表
            subjects: 主题列表，默认为 None，如果为 None 则所有条目使用 "general" 主题
            metadata_list: 元数据列表，默认为 None
            
        Returns:
            成功返回插入的 ID 列表，失败返回 None
        """
        if self.client is None and not self.connect():
            return None
            
        if len(texts) != len(vectors):
            print("错误: 文本列表和向量列表长度不一致")
            return None
            
        try:
            # 准备插入数据
            data = []
            for i, (text, vector) in enumerate(zip(texts, vectors)):
                # 获取主题，如果没有提供则使用默认值 "general"
                subject = "general"
                if subjects is not None and i < len(subjects):
                    subject = subjects[i]
                    
                item = {
                    "id": self._generate_id(),  # 生成唯一ID
                    "vector": vector,
                    "text": text,
                    "subject": subject  # 添加主题字段
                }
                
                # 添加元数据
                if metadata_list is not None and i < len(metadata_list) and isinstance(metadata_list[i], dict):
                    item.update(metadata_list[i])
                
                data.append(item)
            
            # 插入数据
            result = self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            
            if result and 'ids' in result and len(result['ids']) > 0:
                return result['ids']
            return None
        except Exception as e:
            print(f"批量插入数据失败: {str(e)}")
            return None
    
    def search(self, vector, limit=5, filter_expr=None, output_fields=None, include_subdirectories=True):
        """
        根据向量搜索相似内容
        
        Args:
            vector: 查询向量
            limit: 返回结果数量上限，默认为 5
            filter_expr: 过滤表达式，默认为 None
            output_fields: 输出字段列表，默认为 None
            include_subdirectories: 是否包含子目录相关内容，默认为 True
            
        Returns:
            搜索结果列表，失败返回 None
        """
        # @author liangjun
        if self.client is None and not self.connect():
            return None
            
        try:
            # 设置输出字段
            if output_fields is None:
                output_fields = ["text", "subject"]
            elif "subject" not in output_fields:
                output_fields.append("subject")
                
            # 确保包含必要字段
            if "text" not in output_fields:
                output_fields.append("text")
            
            # 设置搜索参数
            search_params = {
                "metric_type": "IP",  # 使用内积作为度量类型
                "params": {"nprobe": 10}  # 搜索探测器数量，可以根据需要调整
            }
            
            print(f"正在使用向量搜索...")
            
            # 直接使用 Milvus 的向量搜索功能
            results = self.client.search(
                collection_name=self.collection_name,
                data=[vector],  # 必须是二维数组
                filter=filter_expr if filter_expr else "",  # 使用 filter 参数而非 expr
                limit=limit,
                output_fields=output_fields,
                search_params=search_params
            )
            
            if not results or len(results) == 0 or len(results[0]) == 0:
                print("没有找到相似的数据")
                return None
                
            print(f"搜索到 {len(results[0])} 条相似数据")
            
            # 处理搜索结果，添加额外字段
            formatted_results = []
            
            # 打印原始结果的详细信息，用于调试
            if results and len(results) > 0 and len(results[0]) > 0:
                print(f"原始搜索结果第一项: {results[0][0]}")
                print(f"原始搜索结果字段: {results[0][0].keys() if isinstance(results[0][0], dict) else 'Not a dict'}")
            
            # 处理每个搜索结果
            for hit in results[0]:
                # 从 Milvus 结果中提取数据
                hit_id = hit.get("id")
                distance = hit.get("distance")  # Milvus 返回的是距离，越小表示越相似
                entity = hit.get("entity", {})  # 实体中包含文本和主题等信息
                
                if hit_id is None:
                    print("跳过无效 ID")
                    continue
                
                # 从 entity 中获取文本和主题
                text = entity.get("text", "")
                subject = entity.get("subject", "general")
                
                # 将距离转换为分数（0-1之间，1表示最相似）
                # 使用简单的转换公式：score = 1 / (1 + distance)
                score = 1.0 / (1.0 + distance) if distance is not None else None
                
                # 构建结果项
                result_item = {
                    "id": hit_id,
                    "score": score,  # 转换后的分数
                    "text": text,  # 从 entity 中获取的文本
                    "subject": subject  # 从 entity 中获取的主题
                }
                
                # 复制其他字段
                for field in output_fields:
                    if field not in ["id", "score", "text", "subject"] and field in entity:
                        result_item[field] = entity.get(field)
                        
                formatted_results.append(result_item)
            
            return [formatted_results]
            
        except Exception as e:
            print(f"搜索数据失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def query(self, filter_expr=None, output_fields=None, limit=100):
        """
        查询数据
        
        Args:
            filter_expr: 过滤表达式，默认为 None
            output_fields: 输出字段列表，默认为 None
            limit: 返回结果数量上限，默认为 100
            
        Returns:
            查询结果列表，失败返回 None
        """
        # @author liangjun
        if self.client is None and not self.connect():
            return None
            
        try:
            if output_fields is None:
                output_fields = ["text", "subject"]
            if "vector" not in output_fields:
                output_fields.append("vector")
                
            print(f"正在查询集合: {self.collection_name}, 输出字段: {output_fields}")
            
            # 设置默认表达式
            if filter_expr is None:
                filter_expr = "id >= 0"  # 使用有效的默认过滤条件
                
            # 使用 filter 参数（而不是 expr）
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,  # 使用 filter 参数
                output_fields=output_fields,
                limit=limit
            )
            
            if results:
                print(f"查询到 {len(results)} 条数据")
                if len(results) > 0:
                    print(f"第一条数据的字段: {list(results[0].keys())}")
            else:
                print("查询结果为空")
                
            return results
        except Exception as e:
            print(f"查询数据失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def delete(self, filter_expr):
        """
        删除数据
        
        Args:
            filter_expr: 过滤表达式，用于指定要删除的数据
            
        Returns:
            成功返回删除的 ID 列表，失败返回 None
        """
        if self.client is None and not self.connect():
            return None
            
        try:
            # 执行删除
            results = self.client.delete(
                collection_name=self.collection_name,
                filter=filter_expr
            )
            
            return results
        except Exception as e:
            print(f"删除数据失败: {str(e)}")
            return None
    
    def delete_by_ids(self, ids):
        """
        根据 ID 列表删除数据
        
        Args:
            ids: ID 列表
            
        Returns:
            成功返回删除的 ID 列表，失败返回 None
        """
        # @author liangjun
        if not ids:
            return []
            
        id_list = ids if isinstance(ids, list) else [ids]
        
        # 分批处理，每批最多处理 100 个 ID
        batch_size = 100
        all_deleted_ids = []
        
        for i in range(0, len(id_list), batch_size):
            batch_ids = id_list[i:i+batch_size]
            try:
                # 直接使用 Milvus 的主键删除功能
                results = self.client.delete(
                    collection_name=self.collection_name,
                    pks=batch_ids  # 直接传递 ID 列表作为主键
                )
                
                if results:
                    all_deleted_ids.extend(batch_ids)
                    print(f"已删除第 {i//batch_size + 1} 批数据，共 {len(batch_ids)} 条")
            except Exception as e:
                print(f"删除第 {i//batch_size + 1} 批数据失败: {str(e)}")
                
        return all_deleted_ids
    
    def count(self):
        """
        获取集合中的数据数量
        
        Returns:
            数据数量，失败返回 0
        """
        if self.client is None and not self.connect():
            return 0
            
        try:
            # 获取集合统计信息
            stats = self.client.get_collection_stats(self.collection_name)
            if stats and 'row_count' in stats:
                return stats['row_count']
            return 0
        except Exception as e:
            print(f"获取数据数量失败: {str(e)}")
            return 0
            
    def clear_collection(self, subject=None):
        """
        清空集合中的数据
        
        Args:
            subject: 要清除的数据主题，如果为 None 则清除所有数据
        
        Returns:
            成功返回 True，失败返回 False
        """
        # @author liangjun
        if self.client is None and not self.connect():
            return False
            
        try:
            if subject is None:
                # 如果没有指定主题，直接清除所有数据
                # 获取所有数据的 ID
                # 获取所有数据的 ID
                results = self.client.query(
                    collection_name=self.collection_name,
                    filter="id >= 0",  # 使用 filter 参数
                    output_fields=["id"],
                    limit=16384  # 添加足够大的限制
                )
                
                if not results or len(results) == 0:
                    print(f"集合 {self.collection_name} 中没有数据")
                    return True
                
                # 收集所有 ID
                ids = [item["id"] for item in results if "id" in item]
                
                if not ids:
                    print(f"集合 {self.collection_name} 中没有有效的 ID")
                    return True
                
                # 根据 ID 删除数据
                self.delete_by_ids(ids)
                print(f"集合 {self.collection_name} 已清空，共删除 {len(ids)} 条数据")
                return True
            else:
                # 如果指定了主题，先获取所有数据，然后在内存中过滤
                # 如果指定了主题，先获取所有数据，然后在内存中过滤
                results = self.client.query(
                    collection_name=self.collection_name,
                    filter="id >= 0",  # 使用 filter 参数
                    output_fields=["id", "subject"],
                    limit=16384  # 添加足够大的限制
                )
                
                if not results or len(results) == 0:
                    print(f"集合 {self.collection_name} 中没有数据")
                    return True
                
                # 在内存中过滤符合条件的 ID
                ids = [item["id"] for item in results if "id" in item and "subject" in item and item["subject"] == subject]
                
                if not ids:
                    print(f"集合 {self.collection_name} 中没有主题为 '{subject}' 的数据")
                    return True
                
                # 根据 ID 删除数据
                self.delete_by_ids(ids)
                print(f"已清除主题为 '{subject}' 的数据，共 {len(ids)} 条")
                return True
        except Exception as e:
            print(f"清空集合失败: {str(e)}")
            return False
    
    def _get_max_id(self):
        """
        获取当前集合中的最大 ID
        
        Returns:
            最大 ID，如果集合为空则返回 0
        """
        try:
            # 查询所有数据，只返回 ID字段
            results = self.client.query(
                collection_name=self.collection_name,
                filter="id >= 0",  # 使用 filter 参数
                output_fields=["id"],
                limit=10000  # 设置足够大的限制
            )
            
            if not results or len(results) == 0:
                return 0
                
            # 找出最大 ID
            max_id = 0
            for item in results:
                if 'id' in item and isinstance(item['id'], int) and item['id'] > max_id:
                    max_id = item['id']
            
            return max_id
        except Exception as e:
            print(f"获取最大 ID 失败: {str(e)}")
            return 0
    
    def _generate_id(self):
        """
        生成唯一ID，基于当前集合中的最大 ID
        
        Returns:
            唯一整数ID
        """
        # 如果已经连接到数据库且集合存在，则获取最大 ID
        if self.client is not None and self.collection_exists():
            max_id = self._get_max_id()
            if max_id > self.last_id:
                self.last_id = max_id
        
        self.last_id += 1
        return self.last_id
        
    def close(self):
        """
        关闭数据库连接
        """
        self.client = None
