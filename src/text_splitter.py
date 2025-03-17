# -*- coding: utf-8 -*-

"""
Markdown 文本切片模块
"""

# @author liangjun

import os
import re
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


class MarkdownSplitter:
    """
    Markdown 文本切片器
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        初始化 Markdown 切片器
        
        Args:
            chunk_size: 每个切片的最大字符数
            chunk_overlap: 切片之间的重叠字符数
        """
        # @author liangjun
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 定义 Markdown 标题分割器
        self.headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
            ("####", "header_4"),
            ("#####", "header_5"),
            ("######", "header_6"),
        ]
        
        # 创建标题分割器
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )
        
        # 创建递归字符分割器（用于进一步分割大块文本）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            keep_separator=True
        )
    
    def read_markdown_file(self, file_path: str) -> str:
        """
        读取 Markdown 文件内容
        
        Args:
            file_path: Markdown 文件路径
            
        Returns:
            文件内容字符串
        """
        # @author liangjun
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def clean_html_tags(self, text: str) -> str:
        """
        清除文本中的 HTML 标签，仅保留文本内容
        
        Args:
            text: 包含 HTML 标签的文本
            
        Returns:
            清除 HTML 标签后的文本
        """
        # @author liangjun
        # 使用 BeautifulSoup 清除 HTML 标签
        try:
            soup = BeautifulSoup(text, 'html.parser')
            clean_text = soup.get_text(separator=' ')
            # 去除多余空格
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            return clean_text
        except Exception as e:
            print(f"清除 HTML 标签失败: {str(e)}")
            return text
    
    def split_text(self, text: str) -> List[Dict[str, Any]]:
        """
        将文本按照标题和段落进行切分
        
        Args:
            text: 要切分的 Markdown 文本
            
        Returns:
            切分后的文本块列表，每个块包含文本内容和元数据
        """
        # @author liangjun
        # 首先按照标题进行切分
        header_splits = self.header_splitter.split_text(text)
        
        # 记录每个标题级别的最后一个标题
        last_headers = {i: None for i in range(1, 7)}
        
        # 进一步切分大块文本
        final_chunks = []
        for split in header_splits:
            # 清除 HTML 标签
            clean_content = self.clean_html_tags(split.page_content)
            
            # 提取元数据
            metadata = {}
            for key, value in split.metadata.items():
                if value:
                    metadata[key] = value
            
            # 确定当前标题级别
            current_level = None
            for i in range(1, 7):
                if f"header_{i}" in metadata and metadata[f"header_{i}"]:
                    current_level = i
                    # 更新当前级别的标题
                    last_headers[i] = metadata[f"header_{i}"]
                    # 清除所有更低级别的标题记录
                    for j in range(i+1, 7):
                        last_headers[j] = None
                    break
            
            # 构建标题路径
            title_path = self._build_title_path(metadata)
            if title_path:
                metadata["title_path"] = title_path
            
            # 构建完整的上下文信息，包括所有父级标题
            parent_context = []
            if current_level:
                # 添加所有父级标题作为上下文
                for i in range(1, current_level):
                    if last_headers[i]:
                        parent_title = self.clean_html_tags(last_headers[i])
                        parent_context.append(parent_title)
            
            # 保存完整标题信息，便于搜索时返回相关内容
            metadata["parent_headers"] = parent_context
            metadata["full_context"] = title_path
            
            # 为每个块添加父级标题的摘要信息
            if parent_context:
                parent_summary = " > ".join(parent_context)
                metadata["parent_summary"] = parent_summary
            
            # 如果文本超过最大块大小，进一步切分
            if len(clean_content) > self.chunk_size:
                sub_chunks = self.text_splitter.split_text(clean_content)
                for i, chunk in enumerate(sub_chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    
                    # 为每个子块添加上下文信息，包括父级标题
                    if "parent_summary" in chunk_metadata:
                        context = f"{chunk_metadata['parent_summary']} > {title_path}\n{chunk}"
                    else:
                        context = f"{title_path}\n{chunk}"
                    chunk_metadata["context"] = context
                    
                    # 移除子目录关系标记，不再按层级切分
                    final_chunks.append({"text": chunk, "metadata": chunk_metadata})
            else:
                metadata["chunk_index"] = 0
                
                # 添加上下文信息，包括父级标题
                if "parent_summary" in metadata:
                    context = f"{metadata['parent_summary']} > {title_path}\n{clean_content}"
                else:
                    context = f"{title_path}\n{clean_content}"
                metadata["context"] = context
                
                # 移除子目录关系标记，不再按层级切分
                final_chunks.append({"text": clean_content, "metadata": metadata})
        
        return final_chunks
    
    def _build_title_path(self, metadata: Dict[str, str]) -> str:
        """
        从元数据构建标题路径
        
        Args:
            metadata: 包含标题信息的元数据
            
        Returns:
            标题路径字符串
        """
        # @author liangjun
        title_parts = []
        for i in range(1, 7):  # 支持 6 级标题
            key = f"header_{i}"
            if key in metadata and metadata[key]:
                # 清除标题中可能存在的 HTML 标签
                clean_title = self.clean_html_tags(metadata[key])
                title_parts.append(clean_title)
        
        return " > ".join(title_parts) if title_parts else ""
    
    def split_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        切分 Markdown 文件
        
        Args:
            file_path: Markdown 文件路径
            
        Returns:
            切分后的文本块列表，每个块包含文本内容和元数据
        """
        # @author liangjun
        try:
            text = self.read_markdown_file(file_path)
            file_name = os.path.basename(file_path)
            chunks = self.split_text(text)
            
            # 添加文件信息到元数据
            for chunk in chunks:
                chunk["metadata"]["source"] = file_path
                chunk["metadata"]["file_name"] = file_name
                
                # 确保每个块都有完整的上下文信息
                if "title_path" in chunk["metadata"] and chunk["metadata"]["title_path"]:
                    # 添加文件名作为最顶层上下文
                    file_context = os.path.splitext(file_name)[0]  # 去除扩展名
                    if not chunk["metadata"]["title_path"].startswith(file_context):
                        title_path = chunk["metadata"]["title_path"]
                        chunk["metadata"]["full_context"] = f"{file_context} > {title_path}"
            
            print(f"成功切分文件，生成 {len(chunks)} 个文本块")
            return chunks
        except Exception as e:
            print(f"切分文件失败: {str(e)}")
            return []
