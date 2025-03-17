#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
命令行交互界面

@author liangjun
"""

import os
import sys
import argparse
from .app import VectorApp

class CLI:
    """
    命令行交互界面
    
    提供交互式的命令行界面，用于插入文本、删除文本向量、搜索文本等功能
    """
    
    def __init__(self, db_path="./data/milvus_db.db", collection_name="text_collection"):
        """
        初始化命令行界面
        
        Args:
            db_path: 数据库存储路径，必须以 .db 结尾
            collection_name: 集合名称
        """
        self.app = VectorApp(db_path, collection_name)
        self.commands = {
            "help": self.show_help,
            "insert": self.insert_text,
            "batchinsert": self.batch_insert_texts,
            "search": self.search_text,
            "delete": self.delete_text,
            "list": self.list_texts,
            "count": self.count_texts,
            "clear": self.clear_all_texts,
            "split": self.split_markdown,
            "chat": self.chat_with_llm,
            "exit": self.exit_app,
            "quit": self.exit_app
        }
    
    def show_help(self):
        """
        显示帮助信息
        """
        print("\n可用命令:")
        print("  help                - 显示帮助信息")
        print("  insert <文本>       - 插入文本到数据库")
        print("  batchInsert         - 批量插入文本到数据库")
        print("  search <查询文本>   - 搜索相似文本")
        print("  delete <ID>         - 根据ID删除文本")
        print("  list                - 列出所有文本")
        print("  count               - 显示文本数量")
        print("  clear               - 清除文本，可指定主题")
        print("  split <文件路径>   - 切片 Markdown 文件并存入数据库")
        print("  chat <问题>         - 使用大模型回答问题")
        print("  exit/quit           - 退出程序\n")
    
    def insert_text(self, text=None):
        """
        插入文本
        
        Args:
            text: 要插入的文本，如果为None则提示用户输入
        """
        # @author liangjun
        if text is None or text.strip() == "":
            text = input("请输入要插入的文本: ").strip()
            if text == "":
                print("错误: 文本不能为空")
                return
        
        # 提示输入主题
        subject = input("请输入文本主题(直接回车使用默认值'general'): ").strip()
        if subject == "":
            subject = "general"
        
        self.app.insert_text(text, subject)
        
    def batch_insert_texts(self):
        """
        批量插入文本
        """
        # @author liangjun
        print("批量插入文本，每行一条文本，输入空行结束")
        
        # 提示输入主题
        subject = input("请输入文本主题(直接回车使用默认值'general'): ").strip()
        if subject == "":
            subject = "general"
        
        texts = []
        subjects = []
        
        print("请输入文本，每行一条，输入空行结束：")
        while True:
            line = input().strip()
            if line == "":
                break
            texts.append(line)
            subjects.append(subject)  # 所有文本使用相同的主题
        
        if not texts:
            print("错误: 没有输入任何文本")
            return
        
        print(f"将插入 {len(texts)} 条文本，主题: {subject}")
        self.app.insert_batch_texts(texts, subjects)
    
    def search_text(self, query=None):
        """
        搜索文本
        
        Args:
            query: 查询文本，如果为None则提示用户输入
        """
        if query is None or query.strip() == "":
            query = input("请输入要搜索的文本: ").strip()
            if query == "":
                print("错误: 查询文本不能为空")
                return
        
        # @author liangjun
        print(f"正在搜索: '{query}'...")
        results = self.app.search_text(query)
        
        # 调试输出
        print(f"搜索结果类型: {type(results)}")
        if results:
            print(f"结果数量: {len(results)}")
            if len(results) > 0:
                print(f"第一个结果类型: {type(results[0])}")
                print(f"第一个结果内容: {results[0]}")
        
        if results and len(results) > 0:
            print("\n搜索结果:")
            for i, hit in enumerate(results):
                # 调试输出当前结果项
                print(f"处理结果 {i+1}:")
                print(hit)
                
                # 安全地获取值，避免对 None 值应用格式化
                score = hit.get('score')
                score_str = f"{score:.4f}" if score is not None else "N/A"
                subject = hit.get('subject', 'general')
                text = hit.get('text', '')
                
                # 显示上下文信息（如果有）
                context_info = ''
                if 'title_path' in hit and hit['title_path'] is not None:
                    context_info += f"路径: {hit['title_path']} | "
                
                # 显示是否为子目录结果
                if hit.get('is_subdirectory', False):
                    context_info += "[子目录] "
                
                print(f"  {i+1}. 相关度: {score_str}, 主题: {subject}\n     {context_info}文本: {text}")
            print()
    
    def delete_text(self, doc_id=None):
        """
        删除文本
        
        Args:
            doc_id: 文档ID，如果为None则提示用户输入
        """
        if doc_id is None or doc_id.strip() == "":
            doc_id = input("请输入要删除的文本ID: ").strip()
            if doc_id == "":
                print("错误: ID不能为空")
                return
        
        try:
            doc_id = int(doc_id)
            self.app.delete_by_id(doc_id)
        except ValueError:
            print("错误: ID必须是整数")
    
    def list_texts(self, subject=None):
        """
        列出文本
        
        Args:
            subject: 按主题筛选，默认为 None 表示列出所有文本
        """
        # @author liangjun
        # 如果命令行没有提供 subject，则提示用户输入
        if subject is None:
            subject_input = input("请输入要筛选的主题(直接回车则列出所有): ").strip()
            if subject_input:
                subject = subject_input
        
        # 查询文本
        texts = self.app.list_all_texts(subject)
        
        # 显示结果
        if texts and len(texts) > 0:
            title = f"\n{subject + ' 主题的' if subject else '所有'}文本 (共 {len(texts)} 条):"
            print(title)
            for doc_id, text, item_subject in texts:
                print(f"  ID: {doc_id}, 主题: {item_subject}, 文本: {text}")
            print()
        else:
            if subject:
                print(f"数据库中没有 '{subject}' 主题的文本\n")
            else:
                print("数据库中没有文本\n")
    
    def count_texts(self):
        """
        显示文本数量
        """
        # @author liangjun
        count = self.app.count()
        print(f"数据库中共有 {count} 条文本\n")
        
    def clear_all_texts(self, subject=None):
        """
        清除文本
        
        Args:
            subject: 要清除的文本主题，如果为 None 则清除所有文本
        """
        # @author liangjun
        
        # 如果没有指定主题，则提示用户输入
        if subject is None or subject.strip() == "":
            subject_input = input("请输入要清除的文本主题(直接回车清除所有文本): ").strip()
            if subject_input != "":
                subject = subject_input
            else:
                subject = None
                
        # 确认操作
        if subject is None:
            confirm = input("警告: 此操作将清除所有文本，确定继续吗？(y/n): ").strip().lower()
        else:
            confirm = input(f"警告: 此操作将清除主题为 '{subject}' 的所有文本，确定继续吗？(y/n): ").strip().lower()
            
        if confirm != 'y':
            print("操作已取消\n")
            return
            
        result = self.app.clear_all(subject)
        if result:
            if subject is None:
                print("所有文本已清除\n")
            else:
                print(f"主题为 '{subject}' 的文本已清除\n")
        else:
            print("清除操作失败\n")
            
    def split_markdown(self, file_path=None):
        """
        切片 Markdown 文件并存入数据库
        
        Args:
            file_path: Markdown 文件路径，如果为 None 则提示用户输入
        """
        # @author liangjun
        
        # 如果没有指定文件路径，则提示用户输入
        if file_path is None or file_path.strip() == "":
            file_path = input("请输入 Markdown 文件路径: ").strip()
            if file_path == "":
                print("错误: 文件路径不能为空")
                return
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件 '{file_path}' 不存在")
            return
            
        # 检查文件类型
        if not file_path.lower().endswith('.md'):
            print("错误: 文件必须是 Markdown (.md) 格式")
            return
            
        # 提示输入主题
        subject = input("请输入文本主题(直接回车使用文件名): ").strip()
        if subject == "":
            subject = None  # 使用文件名作为主题
            
        # 提示输入切片大小和重叠大小
        try:
            chunk_size_input = input("请输入切片大小(直接回车使用默认值 1000): ").strip()
            chunk_size = int(chunk_size_input) if chunk_size_input else 1000
            
            chunk_overlap_input = input("请输入切片重叠大小(直接回车使用默认值 200): ").strip()
            chunk_overlap = int(chunk_overlap_input) if chunk_overlap_input else 200
            
            if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
                print("错误: 切片大小必须大于 0，切片重叠必须大于等于 0 且小于切片大小")
                return
                
        except ValueError:
            print("错误: 请输入有效的数字")
            return
            
        print(f"正在处理 Markdown 文件: {file_path}")
        print(f"切片大小: {chunk_size}, 切片重叠: {chunk_overlap}")
        
        # 处理 Markdown 文件
        try:
            count, ids = self.app.process_markdown_file(file_path, subject, chunk_size, chunk_overlap)
            if count > 0:
                print(f"成功处理 Markdown 文件，共生成 {count} 个切片并存入数据库\n")
            else:
                print("没有生成任何切片，请检查文件内容\n")
        except Exception as e:
            print(f"处理 Markdown 文件失败: {str(e)}\n")
    
    def chat_with_llm(self, query=None):
        """
        与大模型交互，回答问题
        
        Args:
            query: 查询问题，如果为None则提示用户输入
        """
        # @author liangjun
        if query is None or query.strip() == "":
            query = input("请输入您的问题: ").strip()
            if query == "":
                print("错误: 问题不能为空")
                return
        
        # 提示用户设置检索结果数量上限
        limit_input = input("请输入检索结果数量上限(直接回车使用默认值3): ").strip()
        limit = 3  # 默认值
        if limit_input.isdigit():
            limit = int(limit_input)
        
        print(f"正在处理问题: '{query}'...")
        
        # 调用大模型回答问题
        response = self.app.chat(query, limit=limit)
        
        # 显示回答
        print("\n大模型回答:")
        print(response)
        print()
        
        # 询问是否继续对话
        while True:
            continue_chat = input("是否继续对话？(y/n): ").strip().lower()
            if continue_chat == 'n':
                break
            elif continue_chat == 'y':
                follow_up = input("请输入您的问题: ").strip()
                if follow_up == "":
                    print("错误: 问题不能为空")
                    continue
                    
                print(f"正在处理问题: '{follow_up}'...")
                response = self.app.chat(follow_up, limit=limit)
                print("\n大模型回答:")
                print(response)
                print()
            else:
                print("请输入 y 或 n")
    
    def exit_app(self):
        """
        退出程序
        """
        print("正在关闭应用...")
        self.app.close()
        sys.exit(0)
    
    def parse_command(self, command_line):
        """
        解析命令
        
        Args:
            command_line: 命令行字符串
        """
        # @author liangjun
        if not command_line.strip():
            return
            
        parts = command_line.strip().split()
        command = parts[0].lower()
        
        # 特殊处理 list 命令，支持指定 subject
        if command == "list" and len(parts) > 1:
            subject = parts[1]
            self.list_texts(subject)
            return
        
        # 其他命令的处理
        args = None
        if len(parts) > 1:
            # 将除了命令之外的所有部分重新组合为参数
            args = " ".join(parts[1:])
        
        if command in self.commands:
            if args is not None:
                self.commands[command](args)
            else:
                self.commands[command]()
        else:
            print(f"未知命令: {command}")
            self.show_help()
    
    def run(self):
        """
        运行命令行界面
        """
        print("欢迎使用向量数据库交互式应用!")
        print("输入 'help' 查看可用命令\n")
        
        while True:
            try:
                command_line = input("向量应用> ").strip()
                self.parse_command(command_line)
            except KeyboardInterrupt:
                print("\n正在退出...")
                self.app.close()
                break
            except Exception as e:
                print(f"错误: {str(e)}")
