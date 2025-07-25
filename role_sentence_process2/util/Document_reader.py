from haystack import component
from typing import List, Optional, Dict
import os
import logging

logger = logging.getLogger(__name__)

@component
class DocumentReader:
    def __init__(self,
                 recursive: bool = True,
                 ignore_extensions: Optional[List[str]] = None,
                 exclude_dirs: Optional[List[str]] = None,
                 follow_links: bool = False):
        """
        文件内容读取组件

        """
        self.recursive = recursive
        self.ignore_extensions = set(ignore_extensions) if ignore_extensions else None
        self.exclude_dirs = set(exclude_dirs) if exclude_dirs else None
        self.follow_links = follow_links

    @component.output_types(documents=Dict[str, str])  # 修改为输出字典类型
    def run(self, dir_path: str) -> Dict[str, Dict[str, str]]:
        """
        遍历指定目录并返回文件内容字典
        
        :param dir_path: 根目录路径
        :return: 包含文件内容的字典，格式为 {文件名: 文件内容}
        """
        if not os.path.isdir(dir_path):
            logger.error(f"无效的目录路径: {dir_path}")
            return {"documents": {}}

        try:
            # 获取文件路径列表
            file_paths = self._walk_directory(dir_path)
            
            # 读取文件内容并构建字典
            documents = {}
            for file_path in file_paths:
                try:
                    # 获取相对路径作为键
                    rel_path = os.path.relpath(file_path, dir_path)
                    
                    # 处理文件名格式：如果是带下划线的格式，取第三部分；否则直接使用文件名
                    if "_" in rel_path:
                        parts = rel_path.split("_")
                        if len(parts) >= 3:
                            rel_path = parts[2]
                        else:
                            rel_path = os.path.splitext(rel_path)[0]  # 去掉扩展名
                    else:
                        rel_path = os.path.splitext(rel_path)[0]  # 去掉扩展名
                    
                    # 读取文件内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 添加到字典
                    documents[rel_path] = content
                except Exception as e:
                    logger.error(f"读取文件失败: {file_path} - {str(e)}")
            
            logger.info(f"从 {dir_path} 读取了 {len(documents)} 个文件")
            return {"documents": documents}
        except Exception as e:
            logger.error(f"目录遍历失败: {str(e)}")
            return {"documents": {}}

    def _walk_directory(self, root_dir: str) -> List[str]:
        """执行实际的目录遍历（返回完整路径）"""
        collected_files = []
        
        for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=self.follow_links):
            # 处理目录排除逻辑
            dirnames[:] = self._filter_directories(dirpath, dirnames)
            
            # 处理文件收集
            for filename in filenames:
                if self._should_collect_file(filename):
                    full_path = os.path.join(dirpath, filename)
                    collected_files.append(full_path)
            
            # 如果不需要递归，清空子目录列表
            if not self.recursive:
                dirnames.clear()
                
        return collected_files

    def _filter_directories(self, base_dir: str, dirnames: List[str]) -> List[str]:
        """过滤需要处理的目录"""
        filtered = []
        for dirname in dirnames:
            full_path = os.path.join(base_dir, dirname)
            
            # 跳过隐藏目录（以 . 开头）
            if dirname.startswith('.'):
                continue
                
            # 排除指定目录
            if self.exclude_dirs and dirname in self.exclude_dirs:
                logger.debug(f"跳过排除目录: {full_path}")
                continue
                
            filtered.append(dirname)
        return filtered

    def _should_collect_file(self, filename: str) -> bool:
        """判断是否收集该文件"""
        # 跳过隐藏文件
        if filename.startswith('.'):
            return False
            
        # 扩展名过滤
        if self.ignore_extensions:
            ext = os.path.splitext(filename)[1].lower()
            if ext in self.ignore_extensions:
                return False
                
        return True

# # 使用示例
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
    
#     reader = DocumentReader(
#         ignore_extensions=[".tmp", ".bak"],
#         exclude_dirs=["temp"],
#         recursive=True
#     )
    
#     # 运行并获取文件内容字典
#     result = reader.run("/home/daqi/E/rag/genground_xianxiao/dataset")
#     documents = result["documents"]
    
#     # 打印结果
#     print("文件内容字典：")
#     for filename, content in documents.items():
#         print(f"\n文件名: {filename}")
#         print(f"内容预览: {content[:100]}...")  # 只显示前100个字符
    
#     print(f"\n总文件数: {len(documents)}")