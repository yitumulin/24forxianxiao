import json
import re
import logging
from typing import Dict, List, Any
from haystack import component

logger = logging.getLogger(__name__)

@component
class SentenceParser:
    """
    Haystack 组件：解析 LLM 输出的主谓宾三元组JSON列表
    功能：
    1. 提取三元组中的role、Predicate、Object
    2. 移除JSON中的注释
    3. 处理LLM输出的各种不规范JSON格式
    """

    def __init__(self, enable_repair: bool = True):
        self.enable_repair = enable_repair
        self.trailing_comma_pattern = re.compile(r',\s*([\]}])')
        self.missing_quote_pattern = re.compile(r'(\{|\,)\s*([^{\s\'",:]+)\s*:')
        self.single_quote_pattern = re.compile(r"'\s*:\s*'", re.DOTALL)
        self.comment_pattern = re.compile(r'//.*?$', re.MULTILINE)  # 匹配单行注释
    
    @component.output_types(parsed_reply=Dict[str, Any])
    def run(self, llm_replies: List[str], word_max: int, original_sentence: str) -> Dict[str, Any]:
        """
        解析LLM回复为主谓宾三元组
        """
        if not llm_replies:
            return {"parsed_reply": {
                "sentences": [], 
                "isSuccessfully": 0
            }}
        
        llm_reply = llm_replies[0]
        
        try:
            # 从回复中提取真正的JSON部分
            json_array = self.extract_json_array(llm_reply)
            
            if self.enable_repair:
                json_array = self.repair_json_array(json_array)
            
            # 解析JSON数组
            # triplets = self.parse_triplets(json_array)
            sentences = self.parse_sentence(json_array, word_max, original_sentence)

            logger.info(f"成功解析 {len(sentences)} 个句子")

            return {"parsed_reply": {
                "sentences": sentences,
                "isSuccessfully": 1 if sentences!=[] else 0
            }}
        
        except Exception as e:
            logger.error(f"解析过程中发生错误: {str(e)}", exc_info=True)
            return {"parsed_reply": {
                "sentences": [], 
                "isSuccessfully": 0
            }}
    
    def extract_json_array(self, text: str) -> str:
        """从LLM回复中提取JSON数组部分并移除注释"""
        # 首先移除所有注释
        text = self.comment_pattern.sub('', text)
        
        # 尝试找到最外层的JSON数组
        stack = []
        start_index = -1
        
        for i, char in enumerate(text):
            if char == '[':
                if not stack:
                    start_index = i
                stack.append(char)
            elif char == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
                    if not stack:  # 栈为空表示找到完整的数组
                        return text[start_index:i+1]
        
        # 如果括号匹配失败，尝试正则表达式查找
        match = re.search(r'$$.*$$', text, re.DOTALL)
        if match:
            return match.group(0)
        
        # 如果都没有找到，返回空数组字符串
        return "[]"
    
    def repair_json_array(self, json_str: str) -> str:
        """修复JSON数组的常见问题"""
        # 修复尾部逗号
        json_str = self.trailing_comma_pattern.sub(r'\1', json_str)
        
        # 修复单引号
        json_str = self.single_quote_pattern.sub('": "', json_str)
        
        # 修复未加引号的key
        json_str = self.missing_quote_pattern.sub(r'\1"\2":', json_str)
        
        # 添加缺失的引号
        json_str = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', json_str)
        
        return json_str
    
        
    def parse_sentence(self, json_array: str, word_max: int, original_sentence: str) -> List[Dict[str, str]]:
        """解析JSON数组为主谓宾三元组列表"""
        try:
            sentences = json.loads(json_array)
            if not isinstance(sentences, list):
                return []
            
            # 验证和规范化三元组
            valid_sentences = []
            for sentence in sentences:
                if not isinstance(sentence, dict):
                    continue
                
                # 确保有必要的键
                role = sentence.get("role", "")
                behavior_type = sentence.get("behavior_type", "")
                # original_sentence = sentence.get("original_sentence", "")
                original_sentence= original_sentence
                sentence = sentence.get("sentence", "")
                
                
                # 规范化值 - 去除前后空白
                role = role.strip()
                original_sentence = original_sentence.strip()
                sentence = sentence.strip()
                if len(sentence)>word_max:
                    print(original_sentence)
                    print(sentence)
                    return []
                
                # 添加到有效列表
                valid_sentences.append({
                    "role": role,
                    "original_sentence": original_sentence,
                    "sentence": sentence,
                    "behavior_type": behavior_type
                })
            
            return valid_sentences
        
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}\n原始内容: {json_array}")
            return []