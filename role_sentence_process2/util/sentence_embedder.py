import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import Optional, List
import logging

class TextEmbedder:
    """
    文本编码器类，用于对输入文本列表进行嵌入编码
    
    初始化参数:
    model_path: str - 预训练模型路径
    trust_remote_code: bool - 是否信任远程代码执行
    device: Optional[str] - 指定运行设备 (默认为自动选择: 'cuda' 或 'cpu')
    normalize: bool - 是否归一化嵌入向量
    batch_size: int - 批处理大小
    """
    
    def __init__(self,
                 model_path: str = "sentence-transformers/all-MiniLM-L6-v2",
                 trust_remote_code: bool = False,
                 device: Optional[str] = None,
                 normalize: bool = True,
                 batch_size: int = 32):
        
        # 设置模型参数
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.normalize = normalize
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
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
        
        # 获取嵌入维度
        self.embedding_dim = self._get_embedding_dimension()
        
        self.logger.info(f"文本编码模型加载成功: {model_path}")
        self.logger.info(f"模型类型: {'Sentence Transformer' if self.is_sentence_transformer else 'HuggingFace Transformer'}")
        self.logger.info(f"嵌入维度: {self.embedding_dim}")
        self.logger.info(f"运行设备: {self.device}")

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
            self.logger.warning(f"尝试加载Sentence Transformer失败: {str(e)}")
            try:
                model = AutoModel.from_pretrained(
                    self.model_path,
                    trust_remote_code=self.trust_remote_code
                )
                return model, False
            except Exception as e:
                self.logger.error(f"模型加载失败: {str(e)}")
                raise

    def _get_embedding_dimension(self) -> int:
        """获取嵌入向量的维度"""
        # 对于Sentence Transformers模型
        if self.is_sentence_transformer:
            return self.model.get_sentence_embedding_dimension()
        
        # 对于普通Transformer模型
        try:
            return self.model.config.hidden_size
        except AttributeError:
            return 768  # 默认维度

    def encode(self, texts: List[str]) -> List[np.ndarray]:
        """
        对文本列表进行编码
        
        参数:
        texts: List[str] - 要编码的文本列表
        
        返回:
        List[np.ndarray] - 嵌入向量列表
        """
        if not texts:
            self.logger.warning("输入文本列表为空")
            return []
        
        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        self.logger.info(f"开始编码 {len(texts)} 个文本, 共 {total_batches} 个批次")
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            self.logger.debug(f"处理批次: {(i//self.batch_size)+1}/{total_batches}, 文本数: {len(batch)}")
            
            try:
                if self.is_sentence_transformer:
                    batch_embeddings = self._encode_sentence_transformer(batch)
                else:
                    batch_embeddings = self._encode_transformers(batch)
                
                embeddings.extend(batch_embeddings)
            except Exception as e:
                self.logger.error(f"批次 {(i//self.batch_size)+1} 编码失败: {str(e)}")
                # 创建零向量占位符
                batch_embeddings = [np.zeros(self.embedding_dim) for _ in batch]
                embeddings.extend(batch_embeddings)
        
        self.logger.info("文本编码完成")
        return embeddings

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