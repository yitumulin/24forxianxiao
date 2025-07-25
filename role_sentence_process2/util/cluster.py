from haystack import component
from typing import List, Dict, Any
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import logging

logger = logging.getLogger(__name__)

@component
class TextClusterer:
    """
    对词向量列表进行聚类处理，返回聚类结果
    
    输入: 
        - embeddings: 词向量列表 (List[np.ndarray])
        - words: 对应的词列表 (List[str])
        
    输出: 
        - clusters: 聚类结果，包含每个簇的中心点和成员词
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.9,
                 min_cluster_size: int = 2):
        """
        初始化聚类组件
        
        :param similarity_threshold: 余弦相似度阈值，用于确定是否属于同一簇
        :param min_cluster_size: 最小簇大小，小于此值的簇将被视为噪声
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        logger.info(f"聚类组件初始化: 相似度阈值={similarity_threshold}, 最小簇大小={min_cluster_size}")
    
    @component.output_types(clusters=List[Dict[str, Any]])
    def run(self, embeddings: List[np.ndarray], words: List[str]) -> Dict[str, Any]:
        """
        对词向量进行聚类处理
        
        :param embeddings: 词向量列表
        :param words: 对应的词列表
        :return: 聚类结果列表
        """
        # 验证输入
        if not embeddings or not words:
            logger.warning("输入为空，无法进行聚类")
            return {"clusters": []}
        
        if len(embeddings) != len(words):
            logger.error(f"词向量数({len(embeddings)})与词数({len(words)})不匹配")
            return {"clusters": []}
        
        # 转换为numpy数组
        embedding_matrix = np.array(embeddings)
        
        # 计算余弦距离矩阵 (1 - 余弦相似度)
        distance_matrix = pairwise_distances(
            embedding_matrix, 
            metric='cosine'
        )
        
        # 使用DBSCAN进行聚类 (将余弦相似度阈值转换为DBSCAN的eps)
        eps = 1 - self.similarity_threshold
        db = DBSCAN(
            eps=eps, 
            min_samples=self.min_cluster_size,
            metric='precomputed'
        ).fit(distance_matrix)
        
        # 获取聚类标签
        labels = db.labels_
        
        # 统计聚类结果
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        logger.info(f"发现 {n_clusters} 个簇, {n_noise} 个噪声点")
        
        # 组织聚类结果
        clusters = self._organize_clusters(labels, embedding_matrix, words)
        
        return {"clusters": clusters}
    
    def _organize_clusters(self, labels: np.ndarray, 
                          embeddings: np.ndarray, 
                          words: List[str]) -> List[Dict[str, Any]]:
        """
        组织聚类结果，计算簇中心和成员词
        
        :param labels: 聚类标签
        :param embeddings: 词向量矩阵
        :param words: 词列表
        :return: 结构化聚类结果
        """
        unique_labels = set(labels)
        clusters = []
        
        for label in unique_labels:
            # 跳过噪声点
            if label == -1:
                continue
                
            # 获取当前簇的索引
            cluster_indices = np.where(labels == label)[0]
            
            # 获取当前簇的词和向量
            cluster_words = [words[i] for i in cluster_indices]
            cluster_embeddings = embeddings[cluster_indices]
            
            # 计算簇中心 (均值向量)
            cluster_center = np.mean(cluster_embeddings, axis=0)
            
            # 归一化簇中心
            cluster_center_norm = cluster_center / np.linalg.norm(cluster_center)
            
            # 添加到结果
            clusters.append({
                "label": int(label),
                "center": cluster_center_norm.tolist(),
                "words": cluster_words,
                "size": len(cluster_words)
            })
        
        # 按簇大小排序
        clusters.sort(key=lambda x: x["size"], reverse=True)
        return clusters