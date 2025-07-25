from haystack import component
from typing import List, Optional, Dict, Any
import logging
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

@component
class TextListEmbedder:
    """
    对输入的词列表进行编码，返回编码向量列表
    
    输入: 字符串列表 (words: List[str])
    输出: 编码向量列表 (embeddings: List[np.ndarray])
    """
    
    def __init__(self,
                 model_path: str = "sentence-transformers/all-MiniLM-L6-v2",
                 trust_remote_code: bool = False,
                 device: Optional[str] = None,
                 normalize: bool = True,
                 batch_size: int = 32):
        """
        初始化文本编码组件
        :param model_path: 预训练模型路径
        :param trust_remote_code: 是否信任远程代码执行
        :param device: 指定运行设备
        :param normalize: 是否归一化嵌入向量
        :param batch_size: 批处理大小
        """
        # 模型相关设置
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.normalize = normalize
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.model, self.is_sentence_transformer = self._load_model()
        self.model.eval()
        self.model.to(self.device)
        
        # 如果不是Sentence Transformers模型，则加载tokenizer
        if not self.is_sentence_transformer:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=trust_remote_code
            )
            
        logger.info(f"文本编码模型加载成功: {model_path}")
    
    def _load_model(self):
        """加载预训练模型"""
        try:
            model = SentenceTransformer(
                self.model_path, 
                device=self.device,
                trust_remote_code=self.trust_remote_code
            )
            return model, True
        except Exception as e:
            logger.warning(f"尝试加载Sentence Transformer失败: {str(e)}")
            try:
                model = AutoModel.from_pretrained(
                    self.model_path,
                    trust_remote_code=self.trust_remote_code
                )
                return model, False
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                raise

    @component.output_types(embeddings=List[np.ndarray], words=List[str])
    def run(self, words: List[str]) -> Dict[str, Any]:
        """
        核心处理方法
        :param words: 要编码的词列表
        :return: 编码向量列表
        """
        if not words:
            logger.warning("输入词列表为空")
            return {"embeddings": [],  "words": []}
        
        # 生成嵌入向量
        embeddings = self._generate_embeddings(words)
        
        logger.info(f"成功为 {len(words)} 个词生成嵌入向量")
        return {"embeddings": embeddings, "words": words}
    
    def _generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """为文本内容生成嵌入向量"""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            try:
                if self.is_sentence_transformer:
                    batch_embeddings = self._encode_sentence_transformer(batch)
                else:
                    batch_embeddings = self._encode_transformers(batch)
                
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"批次 {i//self.batch_size} 嵌入生成失败: {str(e)}")
                # 为当前批次创建零向量占位符
                batch_dim = self._get_embedding_dimension()
                batch_embeddings = [np.zeros(batch_dim) for _ in batch]
                embeddings.extend(batch_embeddings)
        
        return embeddings

    def _get_embedding_dimension(self) -> int:
        """获取嵌入向量的维度"""
        # 对于Sentence Transformers模型
        if self.is_sentence_transformer:
            return self.model.get_sentence_embedding_dimension()
        
        # 对于普通Transformer模型
        try:
            # 尝试获取隐藏层大小
            return self.model.config.hidden_size
        except AttributeError:
            # 默认维度为768
            return 768

    def _encode_sentence_transformer(self, texts: List[str]) -> List[np.ndarray]:
        """使用Sentence Transformer编码文本"""
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            device=self.device,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )

    def _encode_transformers(self, texts: List[str]) -> List[np.ndarray]:
        """使用普通Transformer模型编码文本"""
        # 准备输入数据
        inputs = self.tokenizer(
            texts, 
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(**inputs)
            
            # 获取最后一层隐藏状态
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            
            # 应用注意力掩码
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # 计算均值池化
            embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            
            # 归一化处理
            if self.normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
            
            return embeddings