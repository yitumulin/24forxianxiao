
"""
基于过滤后的数据（句子数量≥3）重新生成所有统计图表
包括：行为分布图、熵值变化图、语义熵散点图、角色指标分析图等
"""

import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
import matplotlib

# 设置中文字体支持
matplotlib.rc("font", family='WenQuanYi Micro Hei')
plt.rcParams['axes.unicode_minus'] = False

def load_filtered_data(filtered_file):
    """加载过滤后的数据"""
    with open(filtered_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_entropy(behavior_counts):
    """计算基于14类行为类型的语义熵"""
    total = sum(behavior_counts.values())
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in behavior_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def process_json_file_for_entropy(file_path, filtered_roles):
    """处理单个JSON文件，计算每个角色的累积熵值（仅包含过滤后的角色）"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取过滤后的角色名称集合
    filtered_role_names = {role["roleName"] for role in filtered_roles}
    
    # 收集所有非空角色（仅过滤后的）
    unique_roles = set()
    for paragraph in data:
        for sentence in paragraph:
            role = sentence.get('role', '')
            if role and role in filtered_role_names:
                unique_roles.add(role)
    
    # 初始化存储结构
    role_entropy = {role: [] for role in unique_roles}
    role_behavior_counts = {role: defaultdict(int) for role in unique_roles}
    
    # 遍历每个段落
    for para_idx, paragraph in enumerate(data):
        para_role_counts = defaultdict(lambda: defaultdict(int))
        for sentence in paragraph:
            role = sentence.get('role', '')
            if not role or role not in filtered_role_names:
                continue
            b_type = str(sentence.get('behavior_type', '0'))
            para_role_counts[role][b_type] += 1
            role_behavior_counts[role][b_type] += 1
        
        # 计算当前段落每个角色的熵值
        for role in unique_roles:
            if role in para_role_counts:
                entropy_val = calculate_entropy(role_behavior_counts[role])
                role_entropy[role].append(entropy_val)
            else:
                role_entropy[role].append(float('nan'))
    
    return role_entropy, len(data)

def plot_role_entropy_filtered(role_entropy, total_paragraphs, output_file):
    """绘制多角色熵值变化折线图（过滤后版本）"""
    plt.figure(figsize=(14, 8))
    
    color_map = plt.cm.get_cmap('tab20', len(role_entropy))
    
    for idx, (role, data) in enumerate(role_entropy.items()):
        if not data:
            continue
            
        x_vals = [i for i, val in enumerate(data) if not math.isnan(val)]
        y_vals = [val for val in data if not math.isnan(val)]
        
        if not y_vals:
            continue
            
        plt.plot(x_vals, y_vals, 
                 marker='o', 
                 linestyle='-', 
                 linewidth=2,
                 color=color_map(idx),
                 label=role,
                 alpha=0.8)
    
    max_para = max(total_paragraphs, 50)
    xticks = np.arange(0, max_para, 50)
    plt.xticks(xticks)
    
    plt.title('角色语义熵随情节推进的变化 (过滤后)', fontsize=16)
    plt.xlabel('段落序号（每50段标记）', fontsize=12)
    plt.ylabel('累积语义熵（基于14类行为）', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.legend(title='角色列表', 
               bbox_to_anchor=(1.05, 1), 
               loc='upper left',
               framealpha=0.7)
    
    all_vals = []
    for data in role_entropy.values():
        valid_vals = [val for val in data if not math.isnan(val)]
        if valid_vals:
            all_vals.extend(valid_vals)
    
    if all_vals:
        max_entropy = max(all_vals) + 0.2
        plt.ylim(0, max_entropy)
    else:
        plt.ylim(0, 4.0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def process_json_folder_for_entropy(folder_path, output_folder, filtered_roles):
    """处理文件夹中的所有JSON文件，生成熵值变化图（过滤后版本）"""
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                role_entropy, total_paras = process_json_file_for_entropy(file_path, filtered_roles)
                
                if role_entropy:
                    base_name = os.path.splitext(filename)[0]
                    output_file = os.path.join(output_folder, f"{base_name}_multirole_entropy_filtered.png")
                    
                    plot_role_entropy_filtered(role_entropy, total_paras, output_file)
                    print(f"已生成熵值变化图: {output_file}")
                else:
                    print(f"跳过 {filename}（无有效角色数据）")
            except Exception as e:
                print(f"处理失败 {filename}: {str(e)}")

def calculate_behavior_stats_filtered(role_data):
    """计算每个角色类型的行为占比统计量（过滤后版本）"""
    all_behavior_types = set()
    role_type_groups = defaultdict(list)
    
    for role in role_data:
        role_type = role["roleType"]
        total = role["total_sentences"]
        behavior_counts = role["behaviorCounts"]
        
        behavior_ratios = {}
        for b_type, count in behavior_counts.items():
            ratio = count / total
            behavior_ratios[b_type] = ratio
            all_behavior_types.add(b_type)
        
        role_type_groups[role_type].append(behavior_ratios)
    
    sorted_behavior_types = sorted(all_behavior_types, key=lambda x: int(x))
    
    role_stats = {}
    for role_type, ratios_list in role_type_groups.items():
        stats = {}
        for b_type in sorted_behavior_types:
            ratios = [r.get(b_type, 0) for r in ratios_list]
            if ratios:
                mean = np.mean(ratios)
                variance = np.var(ratios)
                stats[b_type] = (mean, variance)
        role_stats[role_type] = stats
    
    return role_stats, sorted_behavior_types

def plot_role_behavior_distribution_filtered(role_type, behavior_stats, behavior_types, output_dir):
    """为单个角色类型绘制行为分布图（过滤后版本）"""
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(behavior_types)))
    
    means = []
    variances = []
    labels = []
    for b_type in behavior_types:
        if b_type in behavior_stats:
            mean, variance = behavior_stats[b_type]
            means.append(mean)
            variances.append(variance)
            labels.append(b_type)
    
    bars = ax.bar(labels, means, color=colors, edgecolor='black', alpha=0.8)
    
    for bar, mean, variance in zip(bars, means, variances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{mean:.2%}\nvar={variance:.4f}",
                ha='center', va='bottom', fontsize=9)
    
    ax.set_title(f"角色类型: {role_type} - 行为分布 (过滤后)", fontsize=16)
    ax.set_xlabel("行为类型（按数值升序排列）", fontsize=12)
    ax.set_ylabel("行为类型占比", fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    safe_role_name = "".join(c for c in role_type if c.isalnum() or c in " -_")
    output_file = os.path.join(output_dir, f"behavior_stats_{safe_role_name}_filtered.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file

def plot_semantic_entropy_filtered(roles_data, output_path):
    """绘制语义熵散点图（过滤后版本）"""
    plt.figure(figsize=(12, 8))
    
    # 按角色类型分组
    role_types = defaultdict(list)
    for role in roles_data:
        role_types[role["roleType"]].append(role)
    
    # 为每个角色类型绘制散点图
    colors = plt.cm.tab10(np.linspace(0, 1, len(role_types)))
    for i, (role_type, roles) in enumerate(role_types.items()):
        entropies = [r["entropy"] for r in roles]
        sentences = [r["total_sentences"] for r in roles]
        
        plt.scatter(sentences, entropies, 
                   label=role_type, 
                   color=colors[i], 
                   alpha=0.7, 
                   s=50)
    
    plt.title('语义熵 vs 句子数分布 (过滤后)', fontsize=16)
    plt.xlabel('句子数', fontsize=12)
    plt.ylabel('语义熵', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_role_metrics_analysis_filtered(roles_data, output_path):
    """绘制角色指标分析图（过滤后版本）"""
    plt.figure(figsize=(15, 10))
    
    # 按角色类型分组
    role_types = defaultdict(list)
    for role in roles_data:
        role_types[role["roleType"]].append(role)
    
    # 计算每个角色类型的平均指标
    metrics = {}
    for role_type, roles in role_types.items():
        avg_entropy = np.mean([r["entropy"] for r in roles])
        avg_sentences = np.mean([r["total_sentences"] for r in roles])
        total_roles = len(roles)
        metrics[role_type] = {
            "avg_entropy": avg_entropy,
            "avg_sentences": avg_sentences,
            "total_roles": total_roles
        }
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 平均熵值对比
    role_names = list(metrics.keys())
    avg_entropies = [metrics[r]["avg_entropy"] for r in role_names]
    bars1 = ax1.bar(role_names, avg_entropies, color='skyblue', alpha=0.7)
    ax1.set_title('各角色类型平均语义熵 (过滤后)', fontsize=14)
    ax1.set_ylabel('平均语义熵')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. 平均句子数对比
    avg_sentences = [metrics[r]["avg_sentences"] for r in role_names]
    bars2 = ax2.bar(role_names, avg_sentences, color='lightcoral', alpha=0.7)
    ax2.set_title('各角色类型平均句子数 (过滤后)', fontsize=14)
    ax2.set_ylabel('平均句子数')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. 角色数量分布
    role_counts = [metrics[r]["total_roles"] for r in role_names]
    bars3 = ax3.bar(role_names, role_counts, color='lightgreen', alpha=0.7)
    ax3.set_title('各角色类型数量分布 (过滤后)', fontsize=14)
    ax3.set_ylabel('角色数量')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 熵值vs句子数散点图
    for role_type, roles in role_types.items():
        entropies = [r["entropy"] for r in roles]
        sentences = [r["total_sentences"] for r in roles]
        ax4.scatter(sentences, entropies, label=role_type, alpha=0.7, s=50)
    
    ax4.set_title('语义熵 vs 句子数分布 (过滤后)', fontsize=14)
    ax4.set_xlabel('句子数')
    ax4.set_ylabel('语义熵')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数：生成所有过滤后的统计图表"""
    print("开始生成所有过滤后的统计图表...")
    
    # 配置路径
    filtered_data_file = "filtered_roles_data.json"
    original_sentences_dir = "../../role_sentence_process2/output/sentences_nor"  # 修改为正确的路径
    entropy_plots_dir = "entropy_plots_filtered"
    behavior_plots_dir = "behavior_stats_plots_filtered"
    
    # 加载过滤后的数据
    filtered_roles = load_filtered_data(filtered_data_file)
    print(f"加载过滤后角色数量: {len(filtered_roles)}")
    
    # 1. 生成熵值变化图
    print("\n1. 生成熵值变化图...")
    process_json_folder_for_entropy(original_sentences_dir, entropy_plots_dir, filtered_roles)
    
    # 2. 生成行为分布图
    print("\n2. 生成行为分布图...")
    os.makedirs(behavior_plots_dir, exist_ok=True)
    role_stats, sorted_behavior_types = calculate_behavior_stats_filtered(filtered_roles)
    
    generated_behavior_files = []
    for role_type, stats in role_stats.items():
        file_path = plot_role_behavior_distribution_filtered(
            role_type, stats, sorted_behavior_types, behavior_plots_dir
        )
        generated_behavior_files.append(file_path)
        print(f"已生成: {file_path}")
    
    # 3. 生成语义熵散点图
    print("\n3. 生成语义熵散点图...")
    plot_semantic_entropy_filtered(filtered_roles, "semantic_entropy_filtered.png")
    
    # 4. 生成角色指标分析图
    print("\n4. 生成角色指标分析图...")
    plot_role_metrics_analysis_filtered(filtered_roles, "role_metrics_analysis_filtered.png")
    
    print(f"\n所有过滤后的统计图表生成完成！")
    print(f"熵值变化图保存在: {entropy_plots_dir}")
    print(f"行为分布图保存在: {behavior_plots_dir}")
    print(f"语义熵散点图: semantic_entropy_filtered.png")
    print(f"角色指标分析图: role_metrics_analysis_filtered.png")

if __name__ == "__main__":
    main() 