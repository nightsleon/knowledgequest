# -*- coding: utf-8 -*-
"""
大模型交互模块

@author liangjun
"""

import os
from typing import List, Dict, Any, Optional

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class LLMChat:
    """
    大模型交互类
    
    提供与大模型交互的功能，包括基于检索结果回答问题
    使用单例模式确保大模型只加载一次
    """
    # 单例实例
    _instance = None
    # 是否已初始化
    _initialized = False
    
    def __new__(cls, model_name="qwen2.5:1.5b"):
        """
        创建或返回单例实例
        
        Args:
            model_name: 模型名称，默认为 qwen2.5:1.5b
        """
        if cls._instance is None:
            cls._instance = super(LLMChat, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_name="qwen2.5:1.5b"):
        """
        初始化大模型交互类
        
        Args:
            model_name: 模型名称，默认为 qwen2.5:1.5b
        """
        # 如果已经初始化过，则直接返回
        if LLMChat._initialized:
            return
            
        self.model_name = model_name
        self.llm = None
        self._init_llm()
        LLMChat._initialized = True
        
    def _init_llm(self):
        """
        初始化大模型
        """
        try:
            self.llm = ChatOllama(model=self.model_name)
            print(f"成功加载模型: {self.model_name}")
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            self.llm = None
    
    def chat_with_context(self, query: str, context_docs: List[Dict[str, Any]]) -> Optional[str]:
        """
        基于检索结果回答问题
        
        Args:
            query: 用户问题
            context_docs: 检索到的相关文档列表
            
        Returns:
            模型回答，失败返回 None
        """
        # @author liangjun
        if self.llm is None:
            print("错误: 大模型未初始化")
            return None
            
        try:
            # 构建上下文
            context_text = "\n\n".join([f"文档 {i+1}: {doc.get('text', '')}" for i, doc in enumerate(context_docs)])
            
            # 构建提示模板
            template = ChatPromptTemplate.from_messages([
                ("system", """
                你是一个智能助手，请基于以下参考文档回答用户的问题。
                如果参考文档中没有相关信息，请直接回答不知道，不要编造信息。
                
                参考文档:
                {context}
                """),
                ("human", "{query}")
            ])
            
            # 创建处理链
            chain = template | self.llm | StrOutputParser()
            
            # 执行链获取回答
            response = chain.invoke({"context": context_text, "query": query})
            
            return response.strip()
            
        except Exception as e:
            print(f"大模型回答失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
