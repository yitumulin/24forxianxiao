
"""
过滤句子数量少于3句的人物
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib

# 设置中文字体支持
matplotlib.rc("font", family='WenQuanYi Micro Hei')
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_and_filter_role_data(folder_path, min_sentences=3):
    """遍历文件夹加载所有JSON文件中的角色数据，并过滤句子数量少于指定值的人物"""
    all_roles = []
    filtered_roles = []
    total_roles = 0
    filtered_count = 0
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_roles.extend(data)
                            # 过滤句子数量少于指定值的人物
                            for role in data:
                                total_roles += 1
                                if role.get("total_sentences", 0) >= min_sentences:
                                    filtered_roles.append(role)
                                else:
                                    filtered_count += 1
                        else:
                            print(f"文件格式错误: {file_path} - 应为JSON数组")
                except Exception as e:
                    print(f"处理文件失败 {file_path}: {str(e)}")
    
    print(f"总角色数量: {total_roles}")
    print(f"过滤后角色数量: {len(filtered_roles)}")
    print(f"被过滤的角色数量: {filtered_count}")
    print(f"过滤条件: 句子数量 >= {min_sentences}")
    
    return filtered_roles

def calculate_filtered_role_metrics(role_data):
    """
    计算过滤后的角色统计指标
    返回: 
        role_metrics = {
            "角色类型": {
                "entropy": (平均值, 方差),
                "count": 角色数量,
                "avg_sentences": 平均句子数
            }
        }
    """
    # 按角色类型分组存储数据
    role_type_data = defaultdict(lambda: {
        "entropy": [],
        "sentences": [],
        "count": 0
    })
    
    # 处理每个角色数据
    for role in role_data:
        role_type = role.get("roleType", "未知类型")
        entropy = role.get("entropy", 0)
        sentences = role.get("total_sentences", 0)
        
        # 存储熵值和句子数，增加计数
        role_type_data[role_type]["entropy"].append(entropy)
        role_type_data[role_type]["sentences"].append(sentences)
        role_type_data[role_type]["count"] += 1
    
    # 计算统计量
    role_metrics = {}
    for role_type, data in role_type_data.items():
        entropy_values = data["entropy"]
        sentences_values = data["sentences"]
        
        role_metrics[role_type] = {
            "entropy": (np.mean(entropy_values), np.var(entropy_values)),
            "count": data["count"],
            "avg_sentences": np.mean(sentences_values)
        }
    
    return role_metrics

def plot_filtered_role_metrics_style1(role_metrics, output_file="filtered_role_metrics_analysis.png"):
    """绘制过滤后的角色指标统计图（style1 - 单柱状图格式）"""
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    
    # 准备数据
    role_types = list(role_metrics.keys())
    entropy_means = [metrics["entropy"][0] for metrics in role_metrics.values()]
    entropy_vars = [metrics["entropy"][1] for metrics in role_metrics.values()]
    role_counts = [metrics["count"] for metrics in role_metrics.values()]
    avg_sentences = [metrics["avg_sentences"] for metrics in role_metrics.values()]
    
    # 设置柱状图位置
    x = np.arange(len(role_types))
    
    # 绘制柱状图
    rects = ax.bar(x, entropy_means, width=0.7, label='角色熵', color='#1f77b4')
    
    # 添加均值和方差标注
    for i, rect in enumerate(rects):
        height = rect.get_height()
        # 添加均值标注（在柱子内部顶部）
        ax.text(rect.get_x() + rect.get_width()/2., height * 0.95,
                f'数量: {role_counts[i]}\n均值: {entropy_means[i]:.4f}\n方差: {entropy_vars[i]:.4f}\n平均句子: {avg_sentences[i]:.1f}',
                ha='center', va='top', fontsize=8, color='white', fontweight='bold')
    
    # 设置图表格式
    ax.set_xlabel('角色类型', fontsize=12)
    ax.set_ylabel('角色熵', fontsize=12)
    ax.set_title('过滤后角色类型与角色熵统计 (句子数≥3)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(role_types, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # 自动调整Y轴范围以容纳标注
    max_value = max(entropy_means) * 1.3
    ax.set_ylim(0, max_value)
    
    # 添加数据标签
    for i, v in enumerate(entropy_means):
        ax.text(i, v + max_value*0.01, f"{v:.4f}", 
                ha='center', va='bottom', fontsize=9)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"图表已保存至: {output_file}")
    plt.close()

def plot_filtered_role_metrics_style2(roles_data, output_path="filtered_role_metrics_analysis1.png"):
    """绘制过滤后的角色指标分析图（style2 - 2x2子图格式）"""
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
    print(f"图表已保存至: {output_path}")

def save_filtered_data(roles_data, output_file="filtered_roles_data.json"):
    """保存过滤后的数据到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(roles_data, f, ensure_ascii=False, indent=2)
    print(f"过滤后的数据已保存至: {output_file}")

def generate_detailed_report(role_metrics, roles_data):
    """生成详细的统计报告"""
    report_file = "filtered_analysis_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("过滤后角色分析报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"过滤条件: 句子数量 >= 3\n")
        f.write(f"总角色数量: {len(roles_data)}\n\n")
        
        f.write("各角色类型详细统计:\n")
        f.write("-" * 50 + "\n")
        
        for role_type, metrics in role_metrics.items():
            f.write(f"\n角色类型: {role_type}\n")
            f.write(f"  角色数量: {metrics['count']}\n")
            f.write(f"  平均熵值: {metrics['entropy'][0]:.4f}\n")
            f.write(f"  熵值方差: {metrics['entropy'][1]:.4f}\n")
            f.write(f"  平均句子数: {metrics['avg_sentences']:.2f}\n")
            
            # 计算该角色类型的角色列表
            type_roles = [r for r in roles_data if r["roleType"] == role_type]
            f.write(f"  角色列表: {[r['roleName'] for r in type_roles]}\n")
        
        # 特别关注忠臣良将和神仙的对比
        f.write("\n\n忠臣良将与神仙对比分析:\n")
        f.write("-" * 50 + "\n")
        
        if "忠臣良将" in role_metrics and "佛道神仙" in role_metrics:
            loyal_metrics = role_metrics["忠臣良将"]
            immortal_metrics = role_metrics["佛道神仙"]
            
            f.write(f"忠臣良将:\n")
            f.write(f"  平均熵值: {loyal_metrics['entropy'][0]:.4f}\n")
            f.write(f"  平均句子数: {loyal_metrics['avg_sentences']:.2f}\n")
            f.write(f"  角色数量: {loyal_metrics['count']}\n")
            
            f.write(f"\n佛道神仙:\n")
            f.write(f"  平均熵值: {immortal_metrics['entropy'][0]:.4f}\n")
            f.write(f"  平均句子数: {immortal_metrics['avg_sentences']:.2f}\n")
            f.write(f"  角色数量: {immortal_metrics['count']}\n")
            
            if loyal_metrics['entropy'][0] < immortal_metrics['entropy'][0]:
                f.write(f"\n结论: 过滤后忠臣良将的熵值({loyal_metrics['entropy'][0]:.4f})仍然低于神仙({immortal_metrics['entropy'][0]:.4f})\n")
            else:
                f.write(f"\n结论: 过滤后忠臣良将的熵值({loyal_metrics['entropy'][0]:.4f})高于神仙({immortal_metrics['entropy'][0]:.4f})\n")
    
    print(f"详细报告已保存至: {report_file}")

if __name__ == "__main__":
    print("开始生成过滤后的角色指标分析...")
    
    # 配置路径
    json_folder = "../output"  # 相对于当前脚本的路径
    
    # 加载并过滤数据
    filtered_roles = load_and_filter_role_data(json_folder, min_sentences=3)
    if not filtered_roles:
        print("未找到有效的角色数据")
        exit()
    
    # 计算过滤后的角色指标
    role_metrics = calculate_filtered_role_metrics(filtered_roles)
    
    # 打印统计结果
    print("\n过滤后角色类型统计结果:")
    for role_type, metrics in role_metrics.items():
        print(f"角色类型: {role_type}")
        print(f"  角色数量: {metrics['count']}")
        print(f"  角色熵: 平均值={metrics['entropy'][0]:.4f}, 方差={metrics['entropy'][1]:.4f}")
        print(f"  平均句子数: {metrics['avg_sentences']:.2f}")
        print("-" * 50)
    
    # 生成两种风格的可视化图表
    print("\n生成过滤后的角色指标分析图...")
    
    # Style1: 单柱状图格式
    plot_filtered_role_metrics_style1(role_metrics, "filtered_role_metrics_analysis.png")
    
    # Style2: 2x2子图格式
    plot_filtered_role_metrics_style2(filtered_roles, "filtered_role_metrics_analysis1.png")
    
    # 保存过滤后的数据
    save_filtered_data(filtered_roles, "filtered_roles_data.json")
    
    # 生成详细报告
    generate_detailed_report(role_metrics, filtered_roles)
    
    print("\n所有过滤后的角色指标分析完成！") 