# -*- coding: utf-8 -*-
import logging
import os
import json
from haystack import Pipeline
from haystack.utils import ComponentDevice
from haystack.components.builders import PromptBuilder
from util.DeepSeek_api import DeepSeekGenerator
from util.Document_reader import DocumentReader
from util.LLm_work_parser import LLMWorkReplyParser
from util.Document_clear import DocumentProcessor
from util.split_text_by_sentence import split_text_by_sentences
from util.role_sentence_parser import SentenceParser
from util.deepseek_iter import generate_intermediate_sentences
from prompt.role_recognation import template as work_template
from prompt.role_sentence_recognation import template as sentence_template
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_process import SentenceProcessor
import threading
import queue

# 创建日志记录器
logger = logging.getLogger()

# 单独设置haystack记录器级别
logging.getLogger("haystack").setLevel(logging.INFO)

# 创建文件输出处理器
file_handler = logging.FileHandler("application_new.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))

# 添加处理器到主记录器
logger.addHandler(file_handler)

# 新添加的文章列表
NEW_ARTICLES = [
    "侯梅英反朝",
    "白玉楼", 
    "兰桥相会",
    "李三娘碾磨",
    "皮箱记(包公案)",
    "秦雪梅吊孝", 
    "王定保借当",
    "珍珠倒卷帘",
    "鞭杆记"
]

# 正确创建设备对象
device = ComponentDevice.from_str("cuda")  # 自动检测可用GPU
input_dir= r"/home/daqi/E/xianxiao/7-14/role_sentence_process2/datset"

# 文档读取
documet_read_pipe = Pipeline()
documet_read_pipe.add_component("reader", DocumentReader(ignore_extensions=[".tmp", ".bak"], exclude_dirs=["temp"], recursive=True))
documet_read_pipe.add_component("document_processor", DocumentProcessor())
documet_read_pipe.connect("reader","document_processor")

# documet_pipe
document_pipe = Pipeline()
document_pipe.add_component("prompt_builder", PromptBuilder(template=work_template))
document_pipe.add_component("json_parser", LLMWorkReplyParser())
document_pipe.add_component("llm", DeepSeekGenerator())
document_pipe.connect("prompt_builder", "llm")
document_pipe.connect("llm.replies", "json_parser")

sentence_pipe = Pipeline()
sentence_pipe.add_component("prompt_builder", PromptBuilder(template=sentence_template))
sentence_pipe.add_component("llm", DeepSeekGenerator())
sentence_pipe.add_component("json_parser", SentenceParser())
sentence_pipe.connect("prompt_builder", "llm")
sentence_pipe.connect("llm.replies", "json_parser")

os.makedirs("output", exist_ok=True)
os.makedirs("output/main_role", exist_ok=True)
os.makedirs("output/sentences_nor", exist_ok=True)
os.makedirs("output/sentences_generation", exist_ok=True)

# 执行
document_dict = documet_read_pipe.run({"reader": {"dir_path": input_dir}})

# 只处理新添加的文章
processed_count = 0
for work_name, whole_content in document_dict["document_processor"]["processed_documents"].items():
    # 检查是否是新的文章
    is_new_article = False
    for new_article in NEW_ARTICLES:
        if new_article in work_name:
            is_new_article = True
            break
    
    if not is_new_article:
        continue
    
    print(f"正在处理新文章: {work_name}")
    processed_count += 1
    
    work_abstract = document_pipe.run({"prompt_builder": {"work_name": work_name,"whole_content": whole_content}})
    
    # 安全获取节点数据
    nodes = work_abstract.get("json_parser", {}).get("parsed_reply", {}).get("nodes", [])
    roles = []
    
    for node in nodes:
        if "Role" not in node.get("labels", []):
            continue
            
        props = node.get("properties", {})
        if "name" in props and "roleType" in props:
            roles.append({
                "roleName": props["name"],
                "roleType": props["roleType"]
            })
    # 保存为角色名称JSON文件
    json_filename = f"output/main_role/{work_name}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(roles, f, ensure_ascii=False, indent=2)

    sentence_dic = split_text_by_sentences(whole_content, context_size=5)
    total_sentences = len(sentence_dic)

    # 创建处理器实例
    processor = SentenceProcessor(total_sentences, work_name, roles)
    # 主执行逻辑修改
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 预分配结果列表（线程安全核心）
        result_list = [None] * len(sentence_dic)  # 固定长度的结果容器
        
        # 提交任务时携带索引和结果列表引用
        futures = []
        for i, text in enumerate(sentence_dic):
            future = executor.submit(
                processor.process_sentence, 
                (i, text),
                result_list  # 传入预分配列表
            )
            futures.append(future)
        
        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"任务失败: {e}")
        
        # 直接获取有序结果
        processor.sentence_processed = [item for item in result_list if item is not None]
    
    # 保存结果
    json_filename = f"output/sentences_nor/{work_name}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(processor.sentence_processed, f, ensure_ascii=False, indent=2)

print(f"新文章处理完成！共处理了 {processed_count} 篇文章") 