from haystack import component
import os
import json
from glob import glob
import ast
from typing import List, Dict, Any, Optional

@component
class FieldExtractor:
    """
    Haystack 组件，用于从指定文件夹中的所有 TXT 文件中提取 JSON 对象列表中指定字段的值。
    """
    def __init__(self, field_to_extract: str = "predicate"):
        """
        初始化字段提取器
        
        参数:
            field_to_extract: 要从每个对象中提取的字段名称，默认为 "predicate"
        """
        self.field_to_extract = field_to_extract
        self.stats = {
            "processed_files": 0,
            "processed_entries": 0,
            "files_with_errors": 0,
            "missing_field_count": 0
        }
    
    @component.output_types(values=List[str], stats=Dict[str, Any])
    def run(self, folder_path: str) -> Dict[str, Any]:
        """
        从指定文件夹中的所有 TXT 文件中提取指定字段的值
        
        参数:
            folder_path: 包含 TXT 文件的文件夹路径
            
        返回:
            dict: 包含以下键的字典:
                - values: 提取的所有字段值列表
                - stats: 处理统计信息
        """
        extracted_values = []
        # 重置统计信息
        self.stats = {
            "processed_files": 0,
            "processed_entries": 0,
            "files_with_errors": 0,
            "missing_field_count": 0
        }
        
        print(f"开始处理文件夹: {folder_path}")
        print(f"提取字段: '{self.field_to_extract}'")
        
        # 遍历文件夹下所有txt文件
        txt_files = glob(os.path.join(folder_path, '*.txt'))
        
        if not txt_files:
            print(f"在文件夹 {folder_path} 中未找到任何txt文件")
            return {
                "values": [], 
                "stats": self.stats
            }
        
        for file_path in txt_files:
            try:
                print(f"处理文件中: {os.path.basename(file_path)}")
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()  # 移除可能的空白字符
                
                # 尝试解析类 JSON 内容
                data_list = None
                # 方法1: 使用ast.literal_eval解析
                try:
                    data_list = ast.literal_eval(content)
                except:
                    # 方法2: 尝试替换单引号为双引号
                    try:
                        json_content = content.replace("'", '"')
                        data_list = json.loads(json_content)
                    except Exception as e:
                        print(f"  解析失败: {str(e)}")
                        self.stats["files_with_errors"] += 1
                        continue
                
                # 确保我们得到的是列表
                if not isinstance(data_list, list):
                    print(f"  文件内容不是列表，跳过")
                    self.stats["files_with_errors"] += 1
                    continue
                    
                file_entries = 0
                file_missing = 0
                # 提取每个对象中的指定字段
                for item in data_list:
                    if isinstance(item, dict) and self.field_to_extract in item:
                        extracted_values.append(item[self.field_to_extract])
                        file_entries += 1
                    elif isinstance(item, dict):
                        # 记录字段缺失的情况
                        file_missing += 1
                
                print(f"  成功提取: {file_entries} 个条目")
                print(f"  缺失字段: {file_missing} 个条目")
                self.stats["processed_files"] += 1
                self.stats["processed_entries"] += file_entries
                self.stats["missing_field_count"] += file_missing
                        
            except Exception as e:
                print(f"  处理文件时出错: {str(e)}")
                self.stats["files_with_errors"] += 1
        
        total_files = len(txt_files)
        print(f"\n处理完成! 共处理 {self.stats['processed_files']}/{total_files} 个文件")
        print(f"提取 {self.stats['processed_entries']} 个值, 缺失字段 {self.stats['missing_field_count']} 个")
        
        return {
            "values": extracted_values, 
            "stats": self.stats
        }