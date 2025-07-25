from haystack import component
from typing import Dict, Any
import json
import pandas as pd
import matplotlib.pyplot as plt
import logging
import numpy as np
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
import matplotlib
matplotlib.rc("font", family='WenQuanYi Micro Hei')

@component
class RoleVisualization:
    """
    Haystack可视化组件：根据角色数据生成包含台词量柱状图和语义熵气泡图的双图分析报告
    
    输入:
     - input_json: JSON格式的角色数据（字典或文件路径）
     - output_path: 输出图像文件路径（可选）
     - dpi: 图像分辨率（可选）
    
    输出: 包含图像数据和统计信息的字典
    """
    
    def __init__(self, dpi: int = 300):
        """
        初始化组件
        :param dpi: 输出图像的分辨率
        """
        self.dpi = dpi
        self._setup_plot_styles()
        logger.info(f"角色可视化组件初始化完成，输出DPI: {dpi}")
    
    def _setup_plot_styles(self):
        """设置绘图样式，确保兼容不同环境"""
        # 尝试使用专业样式
        plt.style.use('ggplot')  # 安全可靠的备选样式
        
        # 设置中文字体和负号显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']  # 多字体备选
        plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
        plt.rcParams['axes.grid'] = True  # 确保有网格线
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['figure.figsize'] = (16, 7)  # 设置图形大小
    
    @component.output_types(
        plot_data=Dict[str, Any],
        stats_data=Dict[str, Any]
    )
    def run(self, input_json: Any, output_path: str = "role_analysis.png") -> Dict[str, Any]:
        """
        处理角色数据并生成可视化结果
        :param input_json: 角色数据 (JSON字符串/文件路径/字典格式)
        :param output_path: 输出图像文件路径
        :return: 包含图像数据和统计信息的字典
        """
        # 1. 解析输入数据
        role_data = self._parse_input(input_json)
        if not role_data:
            logger.error("无效的输入数据")
            return {"plot_data": {}, "stats_data": {}}
        
        logger.info(f"成功加载 {len(role_data)} 个角色的数据")
        
        # 2. 将数据转换为DataFrame
        try:
            df = pd.DataFrame.from_dict(role_data, orient='index').reset_index()
            df.rename(columns={'index': 'role_name'}, inplace=True)
            logger.debug("数据转换为DataFrame成功")
        except Exception as e:
            logger.error(f"数据转换失败: {str(e)}")
            return {"plot_data": {}, "stats_data": {}}
        
        # 3. 计算统计数据
        stats_data = self._calculate_statistics(df)
        
        # 4. 创建可视化图表
        fig = self._create_visualization(df, output_path)
        
        return {
            "plot_data": {
                "figure": fig,
                "output_path": output_path
            },
            "stats_data": stats_data
        }
    
    def _parse_input(self, input_json: Any) -> Dict:
        """解析不同类型的输入为角色数据字典"""
        if isinstance(input_json, dict):
            logger.debug("输入为字典格式")
            return input_json
            
        if isinstance(input_json, str):
            # 尝试解析为JSON字符串
            try:
                return json.loads(input_json)
            except json.JSONDecodeError:
                pass
            
            # 尝试作为文件路径处理
            try:
                path = Path(input_json)
                if path.is_file():
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    logger.error(f"文件未找到: {input_json}")
            except Exception as e:
                logger.error(f"处理输入失败: {str(e)}")
        
        logger.error(f"不支持的输入类型: {type(input_json)}")
        return {}
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """计算基本统计数据"""
        # 按角色类型分组统计
        stats_df = df.groupby('roleType').agg(
            avg_sentences=('sentence_count', 'mean'),
            min_sentences=('sentence_count', 'min'),
            max_sentences=('sentence_count', 'max'),
            avg_entropy=('semantic_entropy', 'mean'),
            min_entropy=('semantic_entropy', 'min'),
            max_entropy=('semantic_entropy', 'max'),
            role_count=('roleType', 'count')
        ).reset_index()
        
        # 计算全局相关性
        corr_value = df['sentence_count'].corr(df['semantic_entropy'])
        
        return {
            "role_stats": stats_df.to_dict(orient='records'),
            "correlation": corr_value,
            "total_roles": len(df),
            "avg_entropy": df['semantic_entropy'].mean(),
            "sentence_stats": {
                "min": df['sentence_count'].min(),
                "max": df['sentence_count'].max(),
                "avg": df['sentence_count'].mean()
            }
        }
    
    def _create_visualization(self, df: pd.DataFrame, output_path: str) -> plt.Figure:
        """创建双图可视化并保存"""
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        # 第一幅图：角色类型与句子数量的关系
        self._create_barplot(df, ax1)
        
        # 第二幅图：语义熵分析
        self._create_scatterplot(df, ax2)
        
        # 添加分析结论标注
        plt.figtext(
            0.5, 
            0.01, 
            "分析结论：语义熵值越高表示角色台词多样性越强，角色气泡越大表示该角色台词数量越多",
            fontsize=12, 
            ha='center',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.7)
        )
        
        # 保存和显示结果
        plt.tight_layout(pad=3.0)
        fig.savefig(output_path, dpi=self.dpi)
        logger.info(f"可视化图表已保存到: {output_path}")
        
        return fig
    
    def _create_barplot(self, df: pd.DataFrame, ax: plt.Axes):
        """创建柱状图"""
        # 手动计算平均值和排序
        role_stats = df.groupby('roleType')['sentence_count'].mean().sort_values(ascending=False)
        role_order = role_stats.index.tolist()
        
        # 为每个角色类型准备数据
        bar_data = []
        for role_type in role_order:
            data_for_role = df[df['roleType'] == role_type]
            bar_data.append({
                "roleType": role_type,
                "mean": data_for_role['sentence_count'].mean(),
                "values": data_for_role['sentence_count'].tolist()
            })
        
        # 创建柱状图
        colors = plt.cm.viridis(np.linspace(0, 1, len(role_order)))
        positions = np.arange(len(role_order))
        
        for i, role_data in enumerate(bar_data):
            ax.bar(
                positions[i], 
                role_data["mean"], 
                color=colors[i],
                alpha=0.8,
                width=0.7
            )
        
        ax.set_title('各角色类型平均台词量', fontsize=15)
        ax.set_xlabel('角色类型', fontsize=12)
        ax.set_ylabel('台词数量', fontsize=12)
        ax.set_xticks(positions)
        ax.set_xticklabels(role_order, rotation=15)
        
        # 添加具体数值标签
        for i, role_data in enumerate(bar_data):
            ax.text(
                positions[i], 
                role_data["mean"] + 0.1, 
                f'{role_data["mean"]:.1f}', 
                ha='center'
            )
    
    def _create_scatterplot(self, df: pd.DataFrame, ax: plt.Axes):
        """创建散点图（气泡图）"""
        # 创建颜色映射
        unique_roles = df['roleType'].unique()
        color_map = {role: plt.cm.tab10(i) for i, role in enumerate(unique_roles)}
        df['color'] = df['roleType'].map(color_map)
        
        # 绘制散点图
        for role_type in unique_roles:
            role_data = df[df['roleType'] == role_type]
            ax.scatter(
                role_data['sentence_count'],
                role_data['semantic_entropy'],
                s=role_data['sentence_count'] * 20,  # 气泡大小表示台词数量
                color=color_map[role_type],
                alpha=0.7,
                label=role_type
            )
        
        ax.set_title('角色台词多样性与语义复杂度', fontsize=15)
        ax.set_xlabel('台词数量', fontsize=12)
        ax.set_ylabel('语义熵 (对话多样性)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 为每个角色添加标签
        for _, row in df.iterrows():
            if row['sentence_count'] > 3 or row['semantic_entropy'] > 0.7:  # 只标注显著角色
                ax.annotate(
                    row['role_name'], 
                    (row['sentence_count'], row['semantic_entropy']),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=9
                )
        
        # 添加图例
        ax.legend(title="角色类型", loc='lower right', frameon=True)

# 使用示例
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 1. 创建组件实例
    visualizer = RoleVisualizationComponent(dpi=200)
    
    # 2. 准备输入数据 (JSON文件路径或JSON字符串)
    # 输入方式1: 直接传递字典数据
    input_data = {
        "曾参": {
            "roleType": "至孝典范",
            "sentences": ["我传真理", "我表贤人", "我啮指"],
            "sentence_count": 3,
            "semantic_entropy": 0.85
        },
        "曾母": {
            "roleType": "天道执行者",
            "sentences": ["我唤儿子", "我接待客人"],
            "sentence_count": 2,
            "semantic_entropy": 0.65
        }
    }
    
    # 输入方式2: JSON文件路径
    # input_data = "role_data.json"
    
    # 输入方式3: JSON字符串
    # input_data = '{"王丁郎": {"roleType": "至孝典范", ...}}'
    
    # 3. 运行组件处理数据
    result = visualizer.run(
        input_json=input_data,
        output_path="output_visualization.png"
    )
    
    # 4. 获取结果
    plot_data = result["plot_data"]
    stats_data = result["stats_data"]
    
    print(f"图表已保存到: {plot_data['output_path']}")
    print("角色统计信息:")
    print(json.dumps(stats_data, indent=2, ensure_ascii=False))
    
    # # 5. 显示图表（可选）
    # plt.show()