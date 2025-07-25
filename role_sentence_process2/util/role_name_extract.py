import os
import json
import ast
import re
from glob import glob
from collections import defaultdict

def process_role_subject_files(folder_path: str) -> dict:
    """
    处理文件夹下所有txt文件，收集每个角色的角色类型和三元组组合成的句子列表
    
    参数:
        folder_path: 包含需要处理的txt文件的文件夹路径
        
    返回:
        dict: 包含以下结构的字典:
            {
                "角色名称1": {
                    "roleType": "角色类型1",
                    "sentences": ["句子1", "句子2", ...]
                },
                "角色名称2": {
                    "roleType": "角色类型2",
                    "sentences": ["句子1", "句子2", ...]
                },
                ...
            }
    """
    # 阶段1：收集全局角色类型映射
    global_role_map = {}
    
    # 最终结果字典
    role_data = defaultdict(lambda: {
        "roleType": "",
        "sentences": []
    })
    
    # 统计信息 (可选，用于监控)
    stats = {
        "total_files": 0,
        "processed_files": 0,
        "files_with_errors": 0,
        "role_types_found": 0,
        "invalid_role_types": 0,
        "triples_processed": 0
    }

    print(f"开始处理文件夹: {folder_path}")
    
    # 获取所有txt文件
    txt_files = glob(os.path.join(folder_path, '*.txt'))
    stats["total_files"] = len(txt_files)
    
    if not txt_files:
        print(f"在文件夹 {folder_path} 中未找到任何txt文件")
        return role_data
    
    # 第一阶段：收集所有角色映射
    print("\n=== 第一阶段: 收集角色类型映射 ===")
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # 解析文件内容
            try:
                # 尝试直接评估为Python字面量
                data = ast.literal_eval(content)
            except Exception as e:
                try:
                    # 尝试作为JSON解析
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # 尝试修复常见的格式问题
                    try:
                        # 处理可能的多行数据
                        fixed_content = content.replace('\n', ',').replace('}{', '},{')
                        if fixed_content.startswith('[') and fixed_content.endswith(']'):
                            data = ast.literal_eval(fixed_content)
                        else:
                            # 可能缺少括号
                            if not fixed_content.startswith('['):
                                fixed_content = '[' + fixed_content
                            if not fixed_content.endswith(']'):
                                fixed_content += ']'
                            data = ast.literal_eval(fixed_content)
                    except:
                        print(f"  文件解析失败: {os.path.basename(file_path)}")
                        stats["files_with_errors"] += 1
                        continue
                except Exception as e:
                    print(f"  文件解析异常: {os.path.basename(file_path)} - {str(e)}")
                    stats["files_with_errors"] += 1
                    continue
            
            if not isinstance(data, list):
                print(f"  文件内容不是列表: {os.path.basename(file_path)}")
                stats["files_with_errors"] += 1
                continue
                
            for item in data:
                if isinstance(item, dict) and "name" in item and "roleType" in item:
                    name = item["name"]
                    role_type = item["roleType"]
                    
                    # 验证角色类型有效性
                    if not role_type or not isinstance(role_type, str):
                        print(f"  无效角色类型: 名称 '{name}'")
                        stats["invalid_role_types"] += 1
                        continue
                    
                    # 添加到全局角色映射
                    global_role_map[name] = role_type
                    stats["role_types_found"] += 1
                    
                    # 添加到最终结果
                    role_data[name]["roleType"] = role_type
                    
        except Exception as e:
            print(f"  处理文件异常: {os.path.basename(file_path)} - {str(e)}")
            stats["files_with_errors"] += 1
    
    print(f"找到 {stats['role_types_found']} 个有效角色类型映射")
    print(f"发现 {stats['invalid_role_types']} 个无效角色类型")
    
    # 第二阶段：处理三元组并创建句子
    print("\n=== 第二阶段: 处理三元组并创建句子 ===")
    for file_path in txt_files:
        try:
            filename = os.path.basename(file_path)
            print(f"处理文件中: {filename}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            try:
                # 尝试直接评估为Python字面量
                data = ast.literal_eval(content)
            except:
                try:
                    # 尝试作为JSON解析
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # 尝试修复格式问题
                    try:
                        fixed_content = content.replace('\n', ',').replace('}{', '},{')
                        if fixed_content.startswith('[') and fixed_content.endswith(']'):
                            data = ast.literal_eval(fixed_content)
                        else:
                            if not fixed_content.startswith('['):
                                fixed_content = '[' + fixed_content
                            if not fixed_content.endswith(']'):
                                fixed_content += ']'
                            data = ast.literal_eval(fixed_content)
                    except:
                        print(f"  文件解析失败: {filename}")
                        stats["files_with_errors"] += 1
                        continue
                except Exception as e:
                    print(f"  文件解析异常: {filename} - {str(e)}")
                    stats["files_with_errors"] += 1
                    continue
            
            if not isinstance(data, list):
                print(f"  文件内容不是列表: {filename}")
                stats["files_with_errors"] += 1
                continue
                
            file_triples = 0
            for item in data:
                if isinstance(item, dict) and "subject" in item and "predicate" in item:
                    subject = item["subject"]
                    predicate = item["predicate"]
                    obj = item.get("object", "")  # 获取object，如果不存在则为空字符串
                    
                    # 创建句子
                    if obj:
                        # 如果object存在：主语 + 谓语 + 宾语
                        sentence = f"我{predicate}{obj}"
                    else:
                        # 如果object不存在：主语 + 谓语
                        sentence = f"我{predicate}"
                    
                    # 如果subject在角色映射中，则添加到该角色的句子列表
                    if subject in global_role_map:
                        # 确保角色已存在
                        if subject in role_data:
                            role_data[subject]["sentences"].append(sentence)
                        else:
                            # 创建新条目
                            role_data[subject] = {
                                "roleType": global_role_map[subject],
                                "sentences": [sentence]
                            }
                        stats["triples_processed"] += 1
                        file_triples += 1
            
            print(f"  提取并转换 {file_triples} 个三元组为句子")
            stats["processed_files"] += 1
            
        except Exception as e:
            print(f"  处理文件异常: {filename} - {str(e)}")
            stats["files_with_errors"] += 1
    
    print(f"处理 {stats['processed_files']} 个文件，转换 {stats['triples_processed']} 个三元组为句子")
    
    # 打印最终统计信息
    print("\n=== 处理完成 ===")
    print(f"发现角色数量: {len(role_data)}")
    print(f"处理文件数: {stats['processed_files']}/{stats['total_files']}")
    print(f"有错误文件数: {stats['files_with_errors']}")
    
    return dict(role_data)  # 将defaultdict转为普通dict

# # 使用方法
# if __name__ == "__main__":
#     input_folder = "/home/daqi/E/rag/xianxiao_down_up/llm_output/triple_671b"
    
#     # 获取角色数据
#     role_sentences_data = process_role_subject_files(input_folder)
    
#     # 示例：打印前3个角色的信息
#     print("\n===== 前3个角色的信息示例 =====")
#     for i, (role_name, data) in enumerate(role_sentences_data.items()):
#         if i >= 3:  # 只展示前3个
#             break
            
#         print(f"\n角色名称: {role_name}")
#         print(f"角色类型: {data['roleType']}")
#         print("相关句子示例:")
#         for j, sentence in enumerate(data["sentences"]):
#             if j < 3:  # 每个角色只展示前3个句子
#                 print(f"  - {sentence}")
    
#     # 保存完整结果到JSON文件
#     output_file = "role_sentences.json"
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(role_sentences_data, f, ensure_ascii=False, indent=2)
#     print(f"\n完整结果已保存到 {output_file}")