from haystack import component
from typing import List, Optional, Dict, Any, Union
import logging
import torch
import json
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

@component
class RoleDataEncoder:
    """
    对角色数据进行编码：读取JSON文件或字典，为每个角色的句子列表生成嵌入向量
    
    输入: JSON文件路径或角色数据字典
    输出: JSON格式字符串 (encoded_data_json: str)
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
    
    def _load_model(self) -> tuple:
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

    @component.output_types(encoded_data_json=str)
    def run(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        核心处理方法
        :param input_data: JSON文件路径或角色数据字典
        :return: JSON格式字符串
        """
        # 1. 读取或获取角色数据
        role_data = self._load_role_data(input_data)
        if not role_data:
            logger.error("无法加载角色数据")
            return {"encoded_data_json": json.dumps({})}
        
        # 2. 获取所有句子列表
        all_sentences = []
        role_sentence_map = []  # 记录句子与角色的映射关系
        
        # 遍历每个角色
        for role_name, role_info in role_data.items():
            sentences = role_info.get("sentences", [])
            # 记录映射：每个句子属于哪个角色
            role_sentence_map.extend([(role_name, sentence) for sentence in sentences])
            all_sentences.extend(sentences)
        
        if not all_sentences:
            logger.warning("未找到任何句子")
            return {"encoded_data_json": json.dumps(role_data, ensure_ascii=False)}
        
        # 3. 为所有句子生成嵌入向量
        embeddings = self._generate_embeddings(all_sentences)
        logger.info(f"成功为 {len(all_sentences)} 个句子生成嵌入向量")
        
        # 4. 重组数据结构，添加嵌入向量
        encoded_data = self._reconstruct_data(role_data, role_sentence_map, embeddings)
        
        # 5. 将结果转换为JSON字符串
        return {"encoded_data_json": json.dumps(encoded_data, ensure_ascii=False)}
    
    def _load_role_data(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """加载角色数据"""
        if isinstance(input_data, str):
            # 处理JSON文件输入
            try:
                with open(input_data, 'r', encoding='utf-8') as f:
                    role_data = json.load(f)
                logger.info(f"成功从 {input_data} 加载数据，包含 {len(role_data)} 个角色")
                return role_data
            except Exception as e:
                logger.error(f"读取JSON文件失败: {str(e)}")
                return {}
        elif isinstance(input_data, dict):
            # 直接使用字典数据
            logger.info(f"使用直接字典输入，包含 {len(input_data)} 个角色")
            return input_data
        else:
            logger.error(f"无效输入类型: {type(input_data)}")
            return {}
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """为文本内容生成嵌入向量"""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            try:
                if self.is_sentence_transformer:
                    batch_embeddings = self._encode_sentence_transformer(batch)
                else:
                    batch_embeddings = self._encode_transformers(batch)
                
                # 将numpy数组转换为列表
                batch_embeddings = [emb.tolist() for emb in batch_embeddings]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"批次 {i//self.batch_size} 嵌入生成失败: {str(e)}")
                # 为当前批次创建零向量占位符
                batch_dim = self._get_embedding_dimension()
                batch_embeddings = [[0.0] * batch_dim for _ in batch]
                embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _reconstruct_data(self, role_data: Dict[str, Any], 
                          role_sentence_map: List[tuple], 
                          embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        重组数据结构，添加嵌入向量
        
        返回格式: {角色名: {角色数据}}
        """
        # 创建角色->嵌入的映射
        role_embeddings = {}
        
        # 初始化每个角色的嵌入列表
        for role_name in role_data.keys():
            role_embeddings[role_name] = []
        
        # 根据映射关系将嵌入分配给正确的角色和句子
        for i, (role_name, _) in enumerate(role_sentence_map):
            if i < len(embeddings):
                role_embeddings[role_name].append(embeddings[i])
        
        # 更新原始数据结构，添加嵌入字段
        for role_name, emb_list in role_embeddings.items():
            if role_name in role_data:
                role_data[role_name]["embeddings"] = emb_list
        
        return role_data

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

# # 使用方法
# if __name__ == "__main__":
#     import logging
#     logging.basicConfig(level=logging.INFO)
    
#     # 创建组件实例
#     encoder = RoleDataEncoder(
#         model_path="/home/daqi/E/rag/haystack/shibing624/text2vec-base-chinese",
#         device="cuda" if torch.cuda.is_available() else "cpu"
#     )
    
#     # 输入数据可以是文件路径或字典
#     # 选项1: 使用文件路径
#     # input_data = "role_sentences.json"
    
#     # 选项2: 直接使用字典数据
#     input_data = {
#         "曾参": {
#             "roleType": "至孝典范",
#             "sentences": ["曾参传真理", "曾参表贤人", "曾参啮指", "曾参痛心"]
#         },
#         "曾母": {
#             "roleType": "天道执行者",
#             "sentences": ["曾母教导曾参", "曾母关爱"]
#         }
#     }
    
#     # 处理数据
#     result = encoder.run(input_data=input_data)
    
#     # 获取JSON字符串结果
#     encoded_data_json = result["encoded_data_json"]
    
#     # 可以直接使用JSON字符串
#     print("JSON字符串结果:")
#     print(encoded_data_json[:200] + "...")  # 打印前200个字符
    
#     # 或者解析为字典
#     encoded_data = json.loads(encoded_data_json)
#     print(f"\n解析为字典后包含 {len(encoded_data)} 个角色")
    
#     # 保存到文件
#     output_json = "role_data_with_embeddings.json"
#     with open(output_json, 'w', encoding='utf-8') as f:
#         f.write(encoded_data_json)
    
#     print(f"处理完成，结果已保存到 {output_json}")