
# from prompt.role_sentence_recognation import template as sentence_template
# from util.role_sentence_parser import SentenceParser
# from haystack.components.builders import PromptBuilder
# from util.DeepSeek_api import DeepSeekGenerator
# from haystack import Pipeline
# import threading
# from tqdm import tqdm  

# sentence_pipe = Pipeline()
# sentence_pipe.add_component("prompt_builder", PromptBuilder(template=sentence_template))
# sentence_pipe.add_component("llm", DeepSeekGenerator())
# sentence_pipe.add_component("json_parser", SentenceParser())
# sentence_pipe.connect("prompt_builder", "llm")
# sentence_pipe.connect("llm.replies", "json_parser")


# class SentenceProcessor:
#     def __init__(self, total_sentences, work_name, roles):
#         self.progress_lock = threading.Lock()
#         self.sentence_processed = []
#         self.processed_count = 0
#         self.roles = roles
        
#         # 创建进度条
#         self.progress_bar = tqdm(
#             total=total_sentences,
#             unit="句",
#             dynamic_ncols=True,
#             desc=f"处理句子 - {work_name}",
#             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
#         )
    
#     def process_sentence(self, args):
#         sentence_index, sentence_dic = args
        
#         before_sentences=sentence_dic["context"]
#         sentence_text = sentence_dic["current_sentence"]
#         print("上文")
#         print(before_sentences)
#         print("当前句子")
#         print(sentence_text)
#         max_retries = 5
#         retry_count = 0
        
#         while True:
#             try:
#                 sentence_result = sentence_pipe.run({
#                     "prompt_builder": {"role_name": self.roles,"before_sentences":before_sentences, "sentences": sentence_text},
#                     "json_parser": {"word_max": 30, "original_sentence": sentence_text}
#                 })
                
#                 json_parser = sentence_result.get("json_parser", {})
#                 parsed_reply = json_parser.get("parsed_reply", {})
                
#                 if parsed_reply.get("isSuccessfully") == 1:
#                     sentences = parsed_reply.get("sentences", [])
#                     break
                    
#             except Exception as e:
#                 print(f"句子处理异常: {e}")
                
#             retry_count += 1
#             if retry_count >= max_retries:
#                 sentences = []
#                 break
        
#         # 线程安全更新
#         with self.progress_lock:
#             self.progress_bar.update(1)
#             self.processed_count += 1
#             self.sentence_processed.extend(sentences)
        
#         return sentences


from prompt.role_sentence_recognation import template as sentence_template
from util.role_sentence_parser import SentenceParser
from haystack.components.builders import PromptBuilder
from util.DeepSeek_api import DeepSeekGenerator
from haystack import Pipeline
import threading
from tqdm import tqdm  

sentence_pipe = Pipeline()
sentence_pipe.add_component("prompt_builder", PromptBuilder(template=sentence_template))
sentence_pipe.add_component("llm", DeepSeekGenerator())
sentence_pipe.add_component("json_parser", SentenceParser())
sentence_pipe.connect("prompt_builder", "llm")
sentence_pipe.connect("llm.replies", "json_parser")


class SentenceProcessor:
    def __init__(self, total_sentences, work_name, roles):
        self.progress_lock = threading.Lock()
        self.sentence_processed = []
        self.processed_count = 0
        self.roles = roles
        
        # 创建进度条
        self.progress_bar = tqdm(
            total=total_sentences,
            unit="句",
            dynamic_ncols=True,
            desc=f"处理句子 - {work_name}",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
    
    def process_sentence(self, args, result_list=None):
        sentence_index, sentence_dic = args
        
        before_sentences=sentence_dic["context"]
        sentence_text = sentence_dic["current_sentence"]
        print("上文")
        print(before_sentences)
        print("当前句子")
        print(sentence_text)
        max_retries = 5
        retry_count = 0
        
        while True:
            try:
                sentence_result = sentence_pipe.run({
                    "prompt_builder": {"role_name": self.roles,"before_sentences":before_sentences, "sentences": sentence_text},
                    "json_parser": {"word_max": 30, "original_sentence": sentence_text}
                })
                
                json_parser = sentence_result.get("json_parser", {})
                parsed_reply = json_parser.get("parsed_reply", {})
                
                if parsed_reply.get("isSuccessfully") == 1:
                    sentences = parsed_reply.get("sentences", [])
                    break
                    
            except Exception as e:
                print(f"句子处理异常: {e}")
                
            retry_count += 1
            if retry_count >= max_retries:
                sentences = []
                break
        
        # 线程安全写入预分配位置
        with self.progress_lock:
            self.progress_bar.update(1)
            if result_list is not None:
                # 关键：直接写入对应索引位置
                result_list[sentence_index] = sentences
        return sentences