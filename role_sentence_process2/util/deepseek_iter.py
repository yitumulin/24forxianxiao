from openai import OpenAI
from typing import List, Dict, Any, Optional
import math
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_intermediate_sentences(
    api_key: str,
    before_sentence: str,
    current_sentence: str,
    after_sentence: str,
    subject: str,
    original_sentence: str,
    model: str = "deepseek-chat",
    api_base_url: str = "https://api.deepseek.com/v1"
) -> Dict[str, Any]:
    """
    使用不同采样参数生成中间句子，并返回包含元数据的完整结果
    
    返回字典结构:
    {
        "input": {
            "subject": 主语,
            "before_sentence": 前句,
            "current_sentence": 当前句,
            "after_sentence": 后句
        },
        "generations": [
            {
                "temperature": 温度值,
                "top_p": top_p值,
                "generated_sentence": 生成句,
                "token_count": token数量,
                "overall_probability": 整体概率,
                "error": 错误信息(如有)
            }
        ]
    }
    """
    # 创建API客户端
    client = OpenAI(
        api_key=api_key,
        base_url=api_base_url,
        timeout=60,
        max_retries=1
    )
    
    # 构建包含前后句的提示词
    prompt = (
        "请根据以下两句话，生成一个合理的中间句子。"
        f"前n个句子: {before_sentence}\n"
        f"后1个句子: {after_sentence}\n"
        f"生成句子的前三个字: {current_sentence[:3  ]}\n"
        "要求：1. 保持原文风格,是一个陈述句，使用提示词中提供生成句子的前三个字 2.生成句子总字数要求不少于9个字，且不多于11个字"
    )
    
    # 定义10种温度(top_p)组合
    param_combinations = [
        (0.2, 0.2), (0.8, 0.8)
    ]
    
    # 创建返回结果结构
    result = {
        "input": {
            "subject": subject,
            "original_sentence": original_sentence,
            "before_sentence": before_sentence,
            "current_sentence": current_sentence,
            "after_sentence": after_sentence
        },
        "generations": []
    }
    
    for i, (temperature, top_p) in enumerate(param_combinations):
        generation = {
            "temperature": temperature,
            "top_p": top_p
        }
        
        try:
            # 执行API调用
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=200,
                logprobs=True,
                top_logprobs=5,
                stream=False
            )
            
            # 解析第一个选择的结果
            choice = response.choices[0]
            
            # 获取生成的句子
            generated_text = choice.message.content
            generation["generated_sentence"] = generated_text
            
            # 计算整体概率
            overall_probability = 1.0
            token_count = 0
            
            # 仅当有logprobs数据时进行计算
            if choice.logprobs and choice.logprobs.content:
                for token_info in choice.logprobs.content:
                    if token_info.logprob is not None:
                        token_prob = math.exp(token_info.logprob)
                        overall_probability *= token_prob
                        token_count += 1
                
                # 处理概率下溢
                if overall_probability == 0.0 and token_count > 0:
                    generation["overall_probability"] = 0.0
                else:
                    generation["overall_probability"] = overall_probability
                
                generation["token_count"] = token_count
            else:
                generation["overall_probability"] = None
                generation["token_count"] = None
                
        except Exception as e:
            error_msg = f"组合#{i+1}生成失败: {str(e)}"
            logger.error(error_msg)
            generation["error"] = error_msg
        
        result["generations"].append(generation)
    
    return result