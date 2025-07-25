import os
import shutil
import json
from collections import defaultdict

def distribute_files_by_linecount(source_dir, output_base_dir):
    """统计JSON文件行数并分配到三个总行数相近的文件夹"""
    # 1. 准备输出目录
    output_dirs = [os.path.join(output_base_dir, f"group_{i+1}") for i in range(3)]
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)

    # 2. 遍历所有JSON文件并统计行数
    file_linecounts = []
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        # 统计行数（支持行分割JSON和标准JSON）
                        if is_line_delimited_json(f):
                            linecount = sum(1 for _ in f)
                        else:
                            data = json.load(f)
                            linecount = len(data) if isinstance(data, list) else 1
                    file_linecounts.append((filepath, linecount))
                except Exception as e:
                    print(f"处理文件失败 {filepath}: {str(e)}")
    
    if not file_linecounts:
        print("未找到JSON文件")
        return
    
    # 3. 按行数降序排序（优化分配均衡性）
    file_linecounts.sort(key=lambda x: x[1], reverse=True)
    
    # 4. 分配文件到三个分组（贪心算法）
    groups = defaultdict(list)
    group_totals = [0, 0, 0]
    
    for filepath, linecount in file_linecounts:
        # 找到当前总行数最小的组
        min_group = group_totals.index(min(group_totals))
        groups[min_group].append((filepath, linecount))
        group_totals[min_group] += linecount
    
    # 5. 移动文件到目标目录
    for group_idx, file_list in groups.items():
        for (src_path, _) in file_list:
            dest_path = os.path.join(output_dirs[group_idx], os.path.basename(src_path))
            shutil.copy2(src_path, dest_path)
    
    # 6. 打印分配结果
    print("\n文件分配完成：")
    for i in range(3):
        print(f"Group {i+1}: {len(groups[i])}个文件, 总行数={group_totals[i]}")
    print(f"最大差值: {max(group_totals) - min(group_totals)}行")

def is_line_delimited_json(file_obj):
    """检测是否为行分割JSON格式"""
    file_obj.seek(0)
    first_char = file_obj.read(1)
    file_obj.seek(0)
    return first_char != '['

# 使用示例
if __name__ == "__main__":
    source_directory = "/home/daqi/E/xianxiao/7-7/role_sentence_process2/output/sentences_nor"  # 替换为JSON文件目录
    output_directory = "./"     # 替换为输出目录
    
    distribute_files_by_linecount(source_directory, output_directory)