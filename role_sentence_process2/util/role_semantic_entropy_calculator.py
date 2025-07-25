from haystack import component
from typing import Dict, Any, List, Optional
import numpy as np
import json
import logging
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
import os
from pathlib import Path
import math

logger = logging.getLogger(__name__)

@component
class RoleSemanticEntropyCalculator:
    """
    计算角色语义熵的Haystack组件
    
    输入: 包含角色数据和嵌入向量的JSON文件路径 或 JSON字符串
    输出: 包含角色语义熵的增强JSON数据
    """
    
    def __init__(self, method: int = 2, sample_size: int = 1000):
        """
        初始化组件
        
        :param method: 熵计算方法 (2: 中心向量法, 3: 聚类轮廓法)
        :param sample_size: 熵计算时的最大样本量
        """
        self.method = method
        self.sample_size = sample_size
        
        # 验证方法参数
        if self.method not in [2, 3]:
            logger.warning(f"无效的方法参数: {method}，将使用默认方法2")
            self.method = 2
            
        logger.info(f"使用方法{self.method}计算语义熵，最大样本量: {sample_size}")
    
    @component.output_types(enhanced_data=Dict[str, Any])
    def run(self, input_json: str) -> Dict[str, Any]:
        """
        核心处理方法
        :param input_json: JSON文件路径 或 直接JSON字符串
        :return: 包含角色语义熵的增强JSON数据
        """
        # 1. 尝试解析JSON数据
        role_data = {}
        try:
            # 尝试作为JSON字符串解析
            role_data = json.loads(input_json)
            logger.info(f"直接解析JSON字符串，包含 {len(role_data)} 个角色")
        except json.JSONDecodeError:
            # 如果解析失败，尝试作为文件路径处理
            try:
                # 检查是否为有效文件路径
                if Path(input_json).is_file():
                    with open(input_json, 'r', encoding='utf-8') as f:
                        role_data = json.load(f)
                    logger.info(f"成功从文件 {input_json} 加载 {len(role_data)} 个角色的数据")
                else:
                    logger.error(f"路径不存在: {input_json}")
                    return {"enhanced_data": {}}
            except Exception as e:
                logger.error(f"处理输入失败: {str(e)}")
                return {"enhanced_data": {}}
        except Exception as e:
            logger.error(f"处理输入失败: {str(e)}")
            return {"enhanced_data": {}}
        
        # 2. 计算每个角色的语义熵
        processed_roles = {}
        total_roles = len(role_data)
        processed_count = 0
        
        for role_name, data in role_data.items():
            try:
                # 提取嵌入向量
                embeddings = data.get("embeddings", [])
                role_type = data.get("roleType", "未知类型")
                sentences = data.get("sentences", [])
                n = len(embeddings)
                
                # 检查是否有足够的数据
                if n < 3:
                    logger.debug(f"跳过角色 {role_name}: 只有 {n} 个句子")
                    continue
                
                # 转换为NumPy数组
                vectors = np.array(embeddings)
                
                # 如果句子太多，进行采样
                if n > self.sample_size:
                    indices = np.random.choice(n, self.sample_size, replace=False)
                    vectors = vectors[indices]
                    n = self.sample_size
                
                # 根据选择的方法计算熵
                if self.method == 2:
                    entropy = self._centroid_method(vectors)
                elif self.method == 3:
                    entropy = self._clustering_method(vectors, n)
                
                # 存储结果
                processed_roles[role_name] = {
                    "roleType": role_type,
                    "sentences": sentences,
                    "sentence_count": len(sentences),
                    "semantic_entropy": entropy,
                    "embeddings_count": n
                }
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"计算角色 {role_name} 的熵时出错: {str(e)}", exc_info=True)
        
        logger.info(f"成功计算 {processed_count}/{total_roles} 个角色的语义熵")
        return {"enhanced_data": processed_roles}
    
    def _centroid_method(self, vectors: np.ndarray) -> float:
        """方法2: 中心向量法计算语义熵"""
        # 计算所有向量的平均值，得到中心向量
        center_vector = np.mean(vectors, axis=0)
        
        # 计算每个向量到中心向量的余弦距离
        distances = []
        for v in vectors:
            # 余弦距离 = 1 - 余弦相似度
            # 添加维度检查避免警告
            if center_vector.ndim == 1 and v.ndim == 1:
                dist = cosine(center_vector, v)
            else:
                dist = cosine(center_vector.flatten(), v.flatten())
            distances.append(dist)
        
        # 平均距离作为熵值
        return np.mean(distances)
    
    # def _clustering_method(self, vectors: np.ndarray, n: int) -> float:
    #     """方法3: 聚类轮廓法计算语义熵"""
    #     if n < 10:
    #         # 对于小样本，使用平均成对距离
    #         return self._pairwise_average_distance(vectors)
    #     else:
    #         # 计算K-means聚类
    #         k = min(5, max(2, n // 2))  # 聚类数量上限5，下限2，每个簇至少有2个点
    #         try:
    #             kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(vectors)
    #             inertia = kmeans.inertia_
                
    #             # 归一化惯性系数作为熵值
    #             # 使用对数转换防止数值过小
    #             normalized_inertia = inertia / (n * np.log(1 + n))
    #             return normalized_inertia
    #         except Exception as e:
    #             logger.error(f"聚类失败: {str(e)}，使用备用方法")
    #             return self._centroid_method(vectors)

    def _clustering_method(self, vectors: np.ndarray, n: int) -> float:
        """方法3: 基于簇大小的语义熵计算"""
        if n < 10:
            # 对于小样本，使用平均成对距离
            return self._pairwise_average_distance(vectors)
        else:
            # 计算K-means聚类
            k = min(3, max(2, n // 2))  # 聚类数量上限5，下限2，每个簇至少有2个点
            try:
                kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(vectors)
                labels = kmeans.labels_
                
                # 计算每个簇的大小
                cluster_counts = np.bincount(labels)
                
                # 计算每个簇的概率（簇大小/总样本数）
                cluster_probs = cluster_counts / n
                
                # 计算基于簇大小的语义熵
                entropy = 0.0
                for p in cluster_probs:
                    if p > 0:  # 避免log(0)错误
                        entropy -= p * math.log(p)
                
                return entropy
            except Exception as e:
                logger.error(f"聚类失败: {str(e)}，使用备用方法")
                return self._centroid_method(vectors)
    
    def _pairwise_average_distance(self, vectors: np.ndarray) -> float:
        """计算所有向量对之间的平均距离（用于小样本）"""
        distances = []
        n = len(vectors)
        
        for i in range(n):
            for j in range(i+1, n):
                # 添加维度检查避免警告
                if vectors[i].ndim == 1 and vectors[j].ndim == 1:
                    dist = cosine(vectors[i], vectors[j])
                else:
                    dist = cosine(vectors[i].flatten(), vectors[j].flatten())
                distances.append(dist)
        
        return np.mean(distances) if distances else 0

# 使用方法
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建组件实例 - 使用方法3计算语义熵
    entropy_calculator = RoleSemanticEntropyCalculator(method=3)
    
    # 方式1: 传递JSON字符串
    json_str = '''
    {
      "曾参": {
        "roleType": "至孝典范",
        "sentences": ["我传真理", "我表贤人", "我啮指"],
        "embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
      }
    }
    '''
    
    # 方式2: 传递文件路径
    # input_json = "role_data_with_embeddings.json"
    
    # 处理数据
    result = entropy_calculator.run(input_json=json_str)
    
    # 获取增强后的数据
    enhanced_data = result["enhanced_data"]
    
    # 保存结果
    output_json = "role_semantic_entropy.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，结果已保存到 {output_json}")
    
    # 示例输出
    print("\n===== 角色熵值结果 =====")
    for role_name, data in enhanced_data.items():
        print(f"角色: {role_name} | 类型: {data['roleType']}")
        print(f"熵值: {data['semantic_entropy']:.4f} | 句子数量: {data['sentence_count']}")
        print(f"示例句子: {data['sentences'][0] if data['sentences'] else '无'}")
        print()