from typing import Dict
from haystack import component
import os
import logging
import re
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)

@component
class DocumentProcessor:
    def __init__(self):
        """初始化文档处理器，移除所有分块相关参数"""
        # 定义空白字符清理正则表达式
        self.whitespace_pattern = r'\s+'
        # 定义括号及括号内容模式（支持中文和英文括号）
        self.bracket_pattern = r'[（$][^）$]*?[\)）]'

    @component.output_types(processed_documents=Dict[str, str])
    def run(self, documents_dict: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """
        处理文档：直接返回作品名称和处理后的全文
        
        :param documents_dict: 输入文档字典 {work_name: content}
        :return: 处理后的文档字典 {work_name: processed_content}
        """
        processed_docs = {}
        total_files = len(documents_dict)
        
        for work_name, content in tqdm(
            documents_dict.items(), 
            total=total_files, 
            desc="处理文档"
        ):
            try:
                # 清理所有括号及括号内内容
                bracket_free_content = self._remove_brackets(content)
                
                # 清理所有空白字符（包括空格、换行等）
                cleaned_content = self._clean_whitespace(bracket_free_content)
                
                # 直接存储处理后的全文
                processed_docs[work_name] = cleaned_content
                
                logger.info(f"已处理作品: {work_name}")
                
            except Exception as e:
                logger.error(f"处理作品 {work_name} 失败: {str(e)}")
                # 失败时保留原始内容
                processed_docs[work_name] = content
                
        return {"processed_documents": processed_docs}

    def _remove_brackets(self, text: str) -> str:
        """
        移除所有括号及其包含的内容
        - 支持中文和英文括号
        """
        return re.sub(self.bracket_pattern, '', text)
    
    def _clean_whitespace(self, text: str) -> str:
        """
        清理所有空白字符
        - 移除所有换行符、空格、制表符等空白字符
        - 使用正则表达式一次性完成所有空白替换
        """
        return re.sub(self.whitespace_pattern, '', text)