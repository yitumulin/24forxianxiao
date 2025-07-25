# import re
# from typing import List

# def split_text_by_sentences(text: str, sentences_per_chunk: int = 10, sentence_overlap: int = 0) -> List[str]:
#     """
#     将长文本按句子分组
    
#     参数:
#     text - 输入的长文本
#     sentences_per_chunk - 每组包含的句子数 (默认20句)
#     sentence_overlap - 组之间的重叠句子数 (默认0句)
    
#     返回:
#     分组后的文本块列表
#     """
#     # 清理文本：移除换行、多余空格和所有双引号
#     cleaned_text = re.sub(r'[""「」"“”]', '', text)  # 移除所有双引号
#     cleaned_text = re.sub(r'\s+', '', cleaned_text.replace('\n', ''))
    
#     # 分割句子 - 使用中文常见结束标点
#     sentence_pattern = r'(?<=[。！？])'
#     sentences = [s for s in re.split(sentence_pattern, cleaned_text) if s]
    
#     # 如果没有句子，返回空列表
#     if not sentences:
#         return []
    
#     chunks = []
#     start = 0
    
#     # 创建句子分组
#     while start < len(sentences):
#         end = min(start + sentences_per_chunk, len(sentences))
#         # 合并组内的句子
#         chunk = ''.join(sentences[start:end])
#         chunks.append(chunk)
        
#         # 移动到下一个分组位置（考虑重叠）
#         start += sentences_per_chunk - sentence_overlap
#         if start >= len(sentences):
#             break
    
#     return chunks

import re
from typing import List, Dict

def split_text_by_sentences(text: str, context_size: int = 3) -> List[Dict[str, str]]:
    """
    将长文本分割为句子，并为每个句子添加上下文
    
    参数:
    text - 输入的长文本
    context_size - 每个句子前包含的上下文句子数 (默认3句)
    
    返回:
    包含字典的列表，每个字典包含:
    - "current_sentence": 当前句子
    - "context": 前n个句子组成的上下文文本
    """
    # 清理文本：移除换行、多余空格和所有双引号
    cleaned_text = re.sub(r'[""「」"“”]', '', text)  # 移除所有双引号
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text.replace('\n', '')).strip()
    
    # 分割句子 - 使用中文常见结束标点
    sentence_pattern = r'(?<=[。！？])'
    sentences = [s.strip() for s in re.split(sentence_pattern, cleaned_text) if s.strip()]
    
    # 如果没有句子，返回空列表
    if not sentences:
        return []
    
    results = []
    
    # 为每个句子创建上下文
    for i in range(len(sentences)):
        # 确定上下文句子的起始位置
        start_idx = max(0, i - context_size)
        
        # 获取上下文句子
        context_sentences = sentences[start_idx:i]
        
        # 创建结果字典
        result_dict = {
            "current_sentence": sentences[i],
            "context": " ".join(context_sentences)  # 将上下文句子连接成字符串
        }
        results.append(result_dict)
    
    return results