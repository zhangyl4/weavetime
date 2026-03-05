#!/usr/bin/env python3
"""
单帧错误案例评估器
评估单帧检索在错误案例上的性能表现
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any


class SingleFrameErrorEvaluator:
    """单帧错误案例评估器"""
    
    def __init__(self, save_dir: str, case_type: str, strategy: str):
        """
        初始化评估器
        
        Args:
            save_dir: 结果保存目录
            case_type: 案例类型
            strategy: 帧选择策略
        """
        self.save_dir = save_dir
        self.case_type = case_type
        self.strategy = strategy
        self.results_file = os.path.join(save_dir, 'results.csv')
        
        # 创建输出目录
        os.makedirs(save_dir, exist_ok=True)
    
    def load_results(self) -> pd.DataFrame:
        """加载结果CSV文件"""
        if not os.path.exists(self.results_file):
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        df = pd.read_csv(self.results_file)
        print(f"Loaded {len(df)} results from {self.results_file}")
        return df
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算评估指标"""
        metrics = {}
        
        # 基本统计
        metrics['total_cases'] = len(df)
        metrics['accuracy'] = df['qa_acc'].mean() if 'qa_acc' in df.columns else 0
        metrics['accuracy_std'] = df['qa_acc'].std() if 'qa_acc' in df.columns else 0
        
        # 原始recall分布
        if 'original_recall' in df.columns:
            metrics['original_recall_mean'] = df['original_recall'].mean()
            metrics['original_recall_std'] = df['original_recall'].std()
            metrics['original_recall_min'] = df['original_recall'].min()
            metrics['original_recall_max'] = df['original_recall'].max()
        
        # 按recall区间分析准确率
        if 'original_recall' in df.columns and 'qa_acc' in df.columns:
            recall_bins = [0, 0.1, 0.3, 0.5, 1.0]
            df['recall_bin'] = pd.cut(df['original_recall'], bins=recall_bins, 
                                    labels=['0-0.1', '0.1-0.3', '0.3-0.5', '0.5-1.0'])
            
            recall_acc = df.groupby('recall_bin')['qa_acc'].agg(['mean', 'std', 'count']).to_dict()
            metrics['accuracy_by_recall'] = recall_acc
        
        # 检索成功率（是否成功选择了帧）
        if 'selected_indices' in df.columns:
            successful_retrievals = df['selected_indices'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
            metrics['retrieval_success_rate'] = successful_retrievals / len(df)
        
        return metrics
    
    def generate_visualizations(self, df: pd.DataFrame, metrics: Dict[str, Any]):
        """生成可视化图表"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Single Frame Error Analysis: {self.case_type} - {self.strategy}', fontsize=16)
        
        # 1. 准确率分布
        if 'qa_acc' in df.columns:
            axes[0, 0].hist(df['qa_acc'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(df['qa_acc'].mean(), color='red', linestyle='--', 
                              label=f'Mean: {df["qa_acc"].mean():.1f}%')
            axes[0, 0].set_xlabel('Accuracy (%)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Accuracy Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 原始recall分布
        if 'original_recall' in df.columns:
            axes[0, 1].hist(df['original_recall'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].axvline(df['original_recall'].mean(), color='red', linestyle='--',
                              label=f'Mean: {df["original_recall"].mean():.3f}')
            axes[0, 1].set_xlabel('Original Recall')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Original Recall Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Recall vs Accuracy散点图
        if 'original_recall' in df.columns and 'qa_acc' in df.columns:
            axes[1, 0].scatter(df['original_recall'], df['qa_acc'], alpha=0.6, color='coral')
            axes[1, 0].set_xlabel('Original Recall')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].set_title('Recall vs Accuracy')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 添加趋势线
            z = np.polyfit(df['original_recall'], df['qa_acc'], 1)
            p = np.poly1d(z)
            axes[1, 0].plot(df['original_recall'], p(df['original_recall']), "r--", alpha=0.8)
        
        # 4. 按recall区间的准确率
        if 'accuracy_by_recall' in metrics and 'mean' in metrics['accuracy_by_recall']:
            recall_means = metrics['accuracy_by_recall']['mean']
            recall_stds = metrics['accuracy_by_recall']['std']
            recall_counts = metrics['accuracy_by_recall']['count']
            
            bins = list(recall_means.keys())
            means = [recall_means.get(bin_name, 0) for bin_name in bins]
            stds = [recall_stds.get(bin_name, 0) for bin_name in bins]
            counts = [recall_counts.get(bin_name, 0) for bin_name in bins]
            
            bars = axes[1, 1].bar(bins, means, yerr=stds, capsize=5, alpha=0.7, color='gold', edgecolor='black')
            axes[1, 1].set_xlabel('Recall Bins')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].set_title('Accuracy by Recall Bins')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 添加样本数量标签
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'n={count}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(self.save_dir, f'{self.case_type}_{self.strategy}_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {plot_path}")
    
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """生成评估报告"""
        report = []
        report.append("="*60)
        report.append("SINGLE FRAME ERROR CASE EVALUATION REPORT")
        report.append("="*60)
        report.append(f"Case Type: {self.case_type}")
        report.append(f"Strategy: {self.strategy}")
        report.append(f"Results Directory: {self.save_dir}")
        report.append("")
        
        # 基本指标
        report.append("Basic Metrics:")
        report.append(f"  Total Cases: {metrics['total_cases']}")
        report.append(f"  Average Accuracy: {metrics['accuracy']:.2f}% (±{metrics['accuracy_std']:.2f}%)")
        
        if 'retrieval_success_rate' in metrics:
            report.append(f"  Retrieval Success Rate: {metrics['retrieval_success_rate']:.2%}")
        
        report.append("")
        
        # 原始recall分析
        if 'original_recall_mean' in metrics:
            report.append("Original Recall Analysis:")
            report.append(f"  Mean Recall: {metrics['original_recall_mean']:.3f} (±{metrics['original_recall_std']:.3f})")
            report.append(f"  Recall Range: {metrics['original_recall_min']:.3f} - {metrics['original_recall_max']:.3f}")
            report.append("")
        
        # 按recall区间的准确率分析
        if 'accuracy_by_recall' in metrics:
            report.append("Accuracy by Recall Bins:")
            acc_by_recall = metrics['accuracy_by_recall']
            if 'mean' in acc_by_recall:
                for bin_name in acc_by_recall['mean'].keys():
                    mean_acc = acc_by_recall['mean'].get(bin_name, 0)
                    std_acc = acc_by_recall['std'].get(bin_name, 0)
                    count = acc_by_recall['count'].get(bin_name, 0)
                    report.append(f"  {bin_name}: {mean_acc:.2f}% (±{std_acc:.2f}%), n={count}")
            report.append("")
        
        # 案例类型特定分析
        if self.case_type == 'no_temporal_overlap':
            report.append("Analysis for No Temporal Overlap Cases:")
            report.append("  These cases originally had 0% recall in temporal windows.")
            report.append("  Single frame retrieval may help by focusing on specific moments.")
        elif self.case_type == 'high_temporal_recall':
            report.append("Analysis for High Temporal Recall Cases:")
            report.append("  These cases originally had >20% recall in temporal windows.")
            report.append("  Single frame retrieval tests if fewer frames can maintain performance.")
        elif self.case_type == 'low_temporal_recall':
            report.append("Analysis for Low Temporal Recall Cases:")
            report.append("  These cases originally had 0-20% recall in temporal windows.")
            report.append("  Single frame retrieval may improve by better frame selection.")
        
        report.append("")
        report.append("="*60)
        
        return "\n".join(report)
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """保存指标到JSON文件"""
        metrics_file = os.path.join(self.save_dir, 'evaluation_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        print(f"Metrics saved to: {metrics_file}")
    
    def run_evaluation(self):
        """运行完整的评估流程"""
        print(f"Starting evaluation for {self.case_type} with {self.strategy} strategy...")
        
        # 加载结果
        df = self.load_results()
        
        # 计算指标
        metrics = self.calculate_metrics(df)
        
        # 生成可视化
        self.generate_visualizations(df, metrics)
        
        # 生成报告
        report = self.generate_report(metrics)
        
        # 保存报告
        report_file = os.path.join(self.save_dir, 'evaluation_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存指标
        self.save_metrics(metrics)
        
        # 打印报告
        print(report)
        
        print(f"\nEvaluation completed. Results saved in: {self.save_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Single Frame Error Case Evaluation')
    parser.add_argument('--save_dir', type=str, required=True, help='Results directory')
    parser.add_argument('--case_type', type=str, required=True, 
                       choices=['no_temporal_overlap', 'high_temporal_recall', 'low_temporal_recall'],
                       help='Error case type')
    parser.add_argument('--strategy', type=str, required=True,
                       choices=['first', 'middle', 'last', 'random'],
                       help='Frame selection strategy')
    
    args = parser.parse_args()
    
    # 创建评估器并运行评估
    evaluator = SingleFrameErrorEvaluator(args.save_dir, args.case_type, args.strategy)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
