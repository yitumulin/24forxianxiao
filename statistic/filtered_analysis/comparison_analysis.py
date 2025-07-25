#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比分析：过滤前后的忠臣良将与神仙熵值对比
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

def load_original_data(folder_path):
    """加载原始数据（未过滤）"""
    all_roles = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_roles.extend(data)
                except Exception as e:
                    print(f"处理文件失败 {file_path}: {str(e)}")
    return all_roles

def load_filtered_data(filtered_file):
    """加载过滤后的数据"""
    with open(filtered_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_comparison_metrics(original_roles, filtered_roles):
    """计算对比指标"""
    # 按角色类型分组
    original_by_type = defaultdict(list)
    filtered_by_type = defaultdict(list)
    
    for role in original_roles:
        original_by_type[role["roleType"]].append(role)
    
    for role in filtered_roles:
        filtered_by_type[role["roleType"]].append(role)
    
    # 计算统计指标
    comparison_data = {}
    
    for role_type in ["忠臣良将", "佛道神仙"]:
        if role_type in original_by_type and role_type in filtered_by_type:
            original_roles_list = original_by_type[role_type]
            filtered_roles_list = filtered_by_type[role_type]
            
            # 原始数据统计
            original_entropies = [r["entropy"] for r in original_roles_list]
            original_sentences = [r["total_sentences"] for r in original_roles_list]
            
            # 过滤后数据统计
            filtered_entropies = [r["entropy"] for r in filtered_roles_list]
            filtered_sentences = [r["total_sentences"] for r in filtered_roles_list]
            
            comparison_data[role_type] = {
                "original": {
                    "count": len(original_roles_list),
                    "avg_entropy": np.mean(original_entropies),
                    "avg_sentences": np.mean(original_sentences),
                    "entropy_std": np.std(original_entropies)
                },
                "filtered": {
                    "count": len(filtered_roles_list),
                    "avg_entropy": np.mean(filtered_entropies),
                    "avg_sentences": np.mean(filtered_sentences),
                    "entropy_std": np.std(filtered_entropies)
                }
            }
    
    return comparison_data

def plot_comparison(comparison_data, output_file="comparison_analysis.png"):
    """绘制对比分析图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    role_types = list(comparison_data.keys())
    x = np.arange(len(role_types))
    width = 0.35
    
    # 1. 熵值对比
    original_entropies = [comparison_data[rt]["original"]["avg_entropy"] for rt in role_types]
    filtered_entropies = [comparison_data[rt]["filtered"]["avg_entropy"] for rt in role_types]
    
    bars1 = ax1.bar(x - width/2, original_entropies, width, label='过滤前', color='lightcoral', alpha=0.7)
    bars2 = ax1.bar(x + width/2, filtered_entropies, width, label='过滤后', color='skyblue', alpha=0.7)
    
    ax1.set_title('忠臣良将与神仙熵值对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('平均熵值')
    ax1.set_xticks(x)
    ax1.set_xticklabels(role_types)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (orig, filt) in enumerate(zip(original_entropies, filtered_entropies)):
        ax1.text(i - width/2, orig + 0.01, f'{orig:.3f}', ha='center', va='bottom')
        ax1.text(i + width/2, filt + 0.01, f'{filt:.3f}', ha='center', va='bottom')
    
    # 2. 句子数对比
    original_sentences = [comparison_data[rt]["original"]["avg_sentences"] for rt in role_types]
    filtered_sentences = [comparison_data[rt]["filtered"]["avg_sentences"] for rt in role_types]
    
    bars3 = ax2.bar(x - width/2, original_sentences, width, label='过滤前', color='lightcoral', alpha=0.7)
    bars4 = ax2.bar(x + width/2, filtered_sentences, width, label='过滤后', color='skyblue', alpha=0.7)
    
    ax2.set_title('忠臣良将与神仙平均句子数对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('平均句子数')
    ax2.set_xticks(x)
    ax2.set_xticklabels(role_types)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (orig, filt) in enumerate(zip(original_sentences, filtered_sentences)):
        ax2.text(i - width/2, orig + 1, f'{orig:.1f}', ha='center', va='bottom')
        ax2.text(i + width/2, filt + 1, f'{filt:.1f}', ha='center', va='bottom')
    
    # 3. 角色数量对比
    original_counts = [comparison_data[rt]["original"]["count"] for rt in role_types]
    filtered_counts = [comparison_data[rt]["filtered"]["count"] for rt in role_types]
    
    bars5 = ax3.bar(x - width/2, original_counts, width, label='过滤前', color='lightcoral', alpha=0.7)
    bars6 = ax3.bar(x + width/2, filtered_counts, width, label='过滤后', color='skyblue', alpha=0.7)
    
    ax3.set_title('忠臣良将与神仙角色数量对比', fontsize=14, fontweight='bold')
    ax3.set_ylabel('角色数量')
    ax3.set_xticks(x)
    ax3.set_xticklabels(role_types)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (orig, filt) in enumerate(zip(original_counts, filtered_counts)):
        ax3.text(i - width/2, orig + 0.5, f'{orig}', ha='center', va='bottom')
        ax3.text(i + width/2, filt + 0.5, f'{filt}', ha='center', va='bottom')
    
    # 4. 熵值差异分析
    entropy_diffs = []
    for rt in role_types:
        original_entropy = comparison_data[rt]["original"]["avg_entropy"]
        filtered_entropy = comparison_data[rt]["filtered"]["avg_entropy"]
        entropy_diffs.append(filtered_entropy - original_entropy)
    
    bars7 = ax4.bar(role_types, entropy_diffs, color=['green' if diff > 0 else 'red' for diff in entropy_diffs], alpha=0.7)
    ax4.set_title('过滤前后熵值变化', fontsize=14, fontweight='bold')
    ax4.set_ylabel('熵值变化 (过滤后 - 过滤前)')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, diff in enumerate(entropy_diffs):
        color = 'green' if diff > 0 else 'red'
        ax4.text(i, diff + (0.01 if diff > 0 else -0.01), f'{diff:+.3f}', 
                ha='center', va='bottom' if diff > 0 else 'top', color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"对比分析图已保存至: {output_file}")

def generate_comparison_report(comparison_data):
    """生成对比分析报告"""
    report_file = "comparison_analysis_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("忠臣良将与神仙熵值对比分析报告\n")
        f.write("=" * 60 + "\n\n")
        f.write("过滤条件: 句子数量 >= 3\n\n")
        
        for role_type in ["忠臣良将", "佛道神仙"]:
            if role_type in comparison_data:
                data = comparison_data[role_type]
                f.write(f"{role_type}详细对比:\n")
                f.write("-" * 40 + "\n")
                
                # 原始数据
                f.write(f"过滤前:\n")
                f.write(f"  角色数量: {data['original']['count']}\n")
                f.write(f"  平均熵值: {data['original']['avg_entropy']:.4f}\n")
                f.write(f"  平均句子数: {data['original']['avg_sentences']:.2f}\n")
                f.write(f"  熵值标准差: {data['original']['entropy_std']:.4f}\n")
                
                # 过滤后数据
                f.write(f"\n过滤后:\n")
                f.write(f"  角色数量: {data['filtered']['count']}\n")
                f.write(f"  平均熵值: {data['filtered']['avg_entropy']:.4f}\n")
                f.write(f"  平均句子数: {data['filtered']['avg_sentences']:.2f}\n")
                f.write(f"  熵值标准差: {data['filtered']['entropy_std']:.4f}\n")
                
                # 变化分析
                entropy_change = data['filtered']['avg_entropy'] - data['original']['avg_entropy']
                sentences_change = data['filtered']['avg_sentences'] - data['original']['avg_sentences']
                count_change = data['filtered']['count'] - data['original']['count']
                
                f.write(f"\n变化分析:\n")
                f.write(f"  熵值变化: {entropy_change:+.4f} ({entropy_change/data['original']['avg_entropy']*100:+.2f}%)\n")
                f.write(f"  句子数变化: {sentences_change:+.2f} ({sentences_change/data['original']['avg_sentences']*100:+.2f}%)\n")
                f.write(f"  角色数量变化: {count_change:+d}\n")
                f.write("\n" + "="*60 + "\n\n")
        
        # 关键发现
        f.write("关键发现:\n")
        f.write("-" * 40 + "\n")
        
        if "忠臣良将" in comparison_data and "佛道神仙" in comparison_data:
            loyal_original = comparison_data["忠臣良将"]["original"]["avg_entropy"]
            loyal_filtered = comparison_data["忠臣良将"]["filtered"]["avg_entropy"]
            immortal_original = comparison_data["佛道神仙"]["original"]["avg_entropy"]
            immortal_filtered = comparison_data["佛道神仙"]["filtered"]["avg_entropy"]
            
            f.write(f"1. 过滤前: 忠臣良将熵值({loyal_original:.4f}) vs 神仙熵值({immortal_original:.4f})\n")
            if loyal_original < immortal_original:
                f.write(f"   结果: 忠臣良将熵值低于神仙\n")
            else:
                f.write(f"   结果: 忠臣良将熵值高于神仙\n")
            
            f.write(f"2. 过滤后: 忠臣良将熵值({loyal_filtered:.4f}) vs 神仙熵值({immortal_filtered:.4f})\n")
            if loyal_filtered < immortal_filtered:
                f.write(f"   结果: 忠臣良将熵值低于神仙\n")
            else:
                f.write(f"   结果: 忠臣良将熵值高于神仙\n")
            
            f.write(f"\n3. 结论: 过滤句子数量少于3句的人物后，")
            if loyal_filtered > immortal_filtered:
                f.write(f"忠臣良将的熵值已经高于神仙，说明句子数量确实是影响熵值的重要因素。\n")
            else:
                f.write(f"忠臣良将的熵值仍然低于神仙，可能需要进一步分析其他因素。\n")
    
    print(f"对比分析报告已保存至: {report_file}")

if __name__ == "__main__":
    print("开始进行过滤前后对比分析...")
    
    # 加载数据
    original_roles = load_original_data("../output")
    filtered_roles = load_filtered_data("filtered_roles_data.json")
    
    print(f"原始角色数量: {len(original_roles)}")
    print(f"过滤后角色数量: {len(filtered_roles)}")
    
    # 计算对比指标
    comparison_data = calculate_comparison_metrics(original_roles, filtered_roles)
    
    # 生成对比图表
    plot_comparison(comparison_data, "comparison_analysis.png")
    
    # 生成对比报告
    generate_comparison_report(comparison_data)
    
    print("对比分析完成！") 