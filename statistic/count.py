import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='WenQuanYi Micro Hei')

def load_json_data(file_path):
    """加载JSON文件数据并处理异常"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"文件加载失败 {file_path}: {str(e)}")
        return None
def merge_and_analyze_files(path1, path2, output_dir, plot_filename="semantic_entropy_plot.png"):
    """合并并分析两个路径下的同名JSON文件，计算语义熵并可视化"""
    import os
    import json
    import math
    from collections import defaultdict
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取两个路径下的JSON文件
    files1 = {f for f in os.listdir(path1) if f.endswith('.json')}
    files2 = {f for f in os.listdir(path2) if f.endswith('.json')}
    common_files = files1 & files2
    
    if not common_files:
        print("未找到同名JSON文件")
        return
    
    # 存储所有角色数据用于可视化
    all_roles_data = []
    
    # 处理每个同名文件
    for filename in common_files:
        # 加载两个文件的数据
        with open(os.path.join(path1, filename), 'r', encoding='utf-8') as f:
            role_data = json.load(f)  # 角色数据
        
        with open(os.path.join(path2, filename), 'r', encoding='utf-8') as f:
            behavior_data = json.load(f)  # 新结构: 列表的列表
        
        if not role_data or not behavior_data:
            continue
        
        # 创建角色名称到类型的映射
        role_mapping = {}
        for item in role_data:
            if 'roleName' in item and 'roleType' in item:
                role_mapping[item['roleName'].strip()] = item['roleType'].strip()
        
        # 统计行为类型数量并计算熵
        stats = defaultdict(lambda: defaultdict(int))
        result = []  # 每个文件单独的结果列表
        
        # 处理新JSON结构: 遍历二维列表
        for sentence_list in behavior_data:  # 第一层: 句子列表
            for item in sentence_list:  # 第二层: 具体条目
                role_name = item.get('role', '').strip()
                behavior_type = item.get('behavior_type')
                
                # 只处理有角色名和有效行为类型的数据
                if role_name in role_mapping and behavior_type is not None:
                    role_type = role_mapping[role_name]
                    stats[(role_name, role_type)][behavior_type] += 1
        
        # 计算每个角色的语义熵
        for (role_name, role_type), behavior_counts in stats.items():
            total_sentences = sum(behavior_counts.values())
            entropy = 0.0
            
            # 计算香农熵 H(X) = -Σ p(x)log₂p(x)
            for count in behavior_counts.values():
                p = count / total_sentences
                if p > 0:  # 避免log(0)的情况
                    entropy -= p * math.log2(p)
            
            # 存储角色数据用于可视化
            all_roles_data.append({
                "roleName": role_name,
                "roleType": role_type,
                "total_sentences": total_sentences,
                "entropy": entropy
            })
            
            # 添加到结果集
            result.append({
                "roleName": role_name,
                "roleType": role_type,
                "behaviorCounts": dict(behavior_counts),
                "total_sentences": total_sentences,
                "entropy": entropy
            })
        
        # 保存分析结果
        output_path = os.path.join(output_dir, f"analysis_{filename}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"分析结果已保存至: {output_path}")
    
    # 可视化：绘制语义熵散点图
    if all_roles_data:
        plot_semantic_entropy(all_roles_data, os.path.join(output_dir, plot_filename))
        print(f"语义熵可视化图已保存至: {os.path.join(output_dir, plot_filename)}")
        
# def merge_and_analyze_files(path1, path2, output_dir, plot_filename="semantic_entropy_plot.png"):
#     """合并并分析两个路径下的同名JSON文件，计算语义熵并可视化"""
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
#     result = []
    
#     # 获取两个路径下的JSON文件
#     files1 = {f for f in os.listdir(path1) if f.endswith('.json')}
#     files2 = {f for f in os.listdir(path2) if f.endswith('.json')}
#     common_files = files1 & files2
    
#     if not common_files:
#         print("未找到同名JSON文件")
#         return
    
#     # 存储所有角色数据用于可视化
#     all_roles_data = []
    
#     # 处理每个同名文件
#     for filename in common_files:
#         # 加载两个文件的数据
#         role_data = load_json_data(os.path.join(path1, filename))
#         behavior_data = load_json_data(os.path.join(path2, filename))
        
#         if not role_data or not behavior_data:
#             continue
        
#         # 创建角色名称到类型的映射
#         role_mapping = {item['roleName'].strip(): item['roleType'].strip() 
#                         for item in role_data if 'roleName' in item}
        
#         # 统计行为类型数量并计算熵
#         stats = defaultdict(lambda: defaultdict(int))
#         role_entropy = {}
        
#         for item in behavior_data:
#             role_name = item.get('role', '').strip()
#             behavior_type = item.get('behavior_type')
            
#             if role_name in role_mapping and behavior_type is not None:
#                 role_type = role_mapping[role_name]
#                 stats[(role_name, role_type)][behavior_type] += 1
        
#         # 计算每个角色的语义熵
#         for (role_name, role_type), behavior_counts in stats.items():
#             total_sentences = sum(behavior_counts.values())
#             entropy = 0.0
            
#             # 计算香农熵 H(X) = -Σ p(x)log₂p(x)
#             for count in behavior_counts.values():
#                 p = count / total_sentences
#                 if p > 0:  # 避免log(0)的情况
#                     entropy -= p * math.log2(p)
            
#             # 存储角色数据用于可视化
#             all_roles_data.append({
#                 "roleName": role_name,
#                 "roleType": role_type,
#                 "total_sentences": total_sentences,
#                 "entropy": entropy
#             })
            
#             # 添加到结果集
#             result.append({
#                 "roleName": role_name,
#                 "roleType": role_type,
#                 "behaviorCounts": dict(behavior_counts),
#                 "total_sentences": total_sentences,
#                 "entropy": entropy
#             })
        
#         # 保存分析结果
#         output_path = os.path.join(output_dir, f"analysis_{filename}")
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(result, f, ensure_ascii=False, indent=2)
#         print(f"分析结果已保存至: {output_path}")
    
#     # 可视化：绘制语义熵散点图
#     if all_roles_data:
#         plot_semantic_entropy(all_roles_data, os.path.join(output_dir, plot_filename))
#         print(f"语义熵可视化图已保存至: {os.path.join(output_dir, plot_filename)}")

def plot_semantic_entropy(roles_data, output_path):
    """绘制角色语义熵散点图"""
    plt.figure(figsize=(12, 8))
    
    # 准备数据
    role_types = list(set(data["roleType"] for data in roles_data))
    color_map = plt.cm.get_cmap('tab20', len(role_types))
    color_dict = {rtype: mcolors.rgb2hex(color_map(i)) for i, rtype in enumerate(role_types)}
    
    # 创建散点图
    for data in roles_data:
        plt.scatter(
            data["total_sentences"],
            data["entropy"],
            s=100,  # 点的大小
            alpha=0.7,
            c=[color_dict[data["roleType"]]],
            edgecolors='w',
            linewidth=0.5
        )
    
    # 添加标签和标题
    plt.xlabel('句子数量', fontsize=14)
    plt.ylabel('语义熵', fontsize=14)
    plt.title('角色行为语义熵分布', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # 创建图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=color_dict[rtype], markersize=10, label=rtype)
        for rtype in role_types
    ]
    plt.legend(handles=legend_elements, title='角色类型', loc='best', fontsize=9)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    path1="/home/daqi/E/xianxiao/7-7/role_sentence_process2/output/main_role"
    path2="/home/daqi/E/xianxiao/7-7/role_sentence_process2/output/sentences_nor"
    output_dir="/home/daqi/E/xianxiao/7-7/statistic/output"
    
    # 执行分析
    merge_and_analyze_files(path1, path2, output_dir)
    print("处理完成！可在输出目录查看分析结果")