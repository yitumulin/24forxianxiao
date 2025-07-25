from haystack import component
from typing import Any, Dict, List, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletion
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@component
class DeepSeekGenerator:
    def __init__(
        self,
        api_key: str = "sk-7feb014eda5b48b1988d7599802f3590",
        api_base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-coder", #reasoner/chat
        timeout: int = 60,
        max_retries: int = 1,  
    ):
        """
        DeepSeek API 生成组件
        
        参数:
        api_key (str): DeepSeek API密钥
        api_base_url (str): API基础URL
        model (str): 模型名称
        timeout (int): 请求超时时间（秒）
        max_retries (int): 最大重试次数
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base_url,
            timeout=timeout,
            max_retries=max_retries
        )
        self.model = model

    def _build_messages(self, content: str) -> List[Dict[str, str]]:
        """自动构建消息结构"""
        return [{
            "role": "user",
            "content": content
        }]

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(self, content: str) -> Dict[str, Any]:  # 参数改为直接接收内容
        """
        执行API调用
        
        参数:
        content (str): 用户输入内容，自动构建消息结构
        
        返回:
        Dict[str, Any]: 包含replies和meta的字典
        """
        try:
            # 自动构建消息
            messages = self._build_messages(content)
            
            # 验证输入内容
            if not isinstance(content, str) or len(content.strip()) == 0:
                raise ValueError("content必须是非空字符串")

            # 执行API调用
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=5000,
                stream=False
            )
            
            return {
                "replies": [choice.message.content for choice in response.choices],
                "meta": [{
                    "id": response.id,
                    "model": response.model,
                    "usage": dict(response.usage) if response.usage else {},
                    "created": response.created
                }]
            }
            
        except Exception as e:
            logger.error(f"API调用失败: {str(e)}")
            return {
                "replies": [],
                "meta": [{"error": str(e)}]
            }
        
if __name__ == "__main__":
    generator = DeepSeekGenerator()
    # 现在只需直接传入内容
    result = generator.run(content="什么是mcp")
    
    if result["replies"]:
        print("生成结果：")
        print(result["replies"][0])
        print("\n元数据：")
        print(result["meta"][0])
    else:
        print("请求失败，错误信息：")
        print(result["meta"][0].get("error", "未知错误"))