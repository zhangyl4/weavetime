import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from logzero import logger


class RetrievalAnalyzer:
    """
    分析retrieval frame记录的工具类
    """
    
    def __init__(self, results_csv_path: str = None, results_data: List[Dict] = None):
        """
        初始化分析器
        
        Args:
            results_csv_path: 结果CSV文件路径
            results_data: 直接传入的结果数据
        """
        if results_csv_path:
            self.df = pd.read_csv(results_csv_path)
            
            # 检查是否存在 retrieval_records 字段
            if 'retrieval_records' in self.df.columns:
                # 安全地解析JSON字符串字段
                def safe_parse_retrieval_records(x):
                    if pd.isna(x) or x == '' or x == '{}':
                        return {}
                    if isinstance(x, dict):
                        return x
                    if isinstance(x, str):
                        # 尝试使用 eval 解析字典字符串（更安全的方式）
                        try:
                            # 先尝试 json.loads
                            return json.loads(x)
                        except (json.JSONDecodeError, ValueError):
                            try:
                                # 如果是 Python 字典格式的字符串，使用 ast.literal_eval
                                import ast
                                return ast.literal_eval(x)
                            except (ValueError, SyntaxError):
                                logger.warning(f"无法解析 retrieval_records: {str(x)[:100]}...")
                                return {}
                    else:
                        return {}
                
                self.df['retrieval_records'] = self.df['retrieval_records'].apply(safe_parse_retrieval_records)
            else:
                logger.warning("CSV文件中没有找到 'retrieval_records' 字段，创建空字段")
                self.df['retrieval_records'] = [{}] * len(self.df)
                
            # 检查是否存在 similarity_scores 字段
            if 'similarity_scores' in self.df.columns:
                def safe_parse_similarity_scores(x):
                    if pd.isna(x) or x == '' or x == '{}':
                        return {}
                    if isinstance(x, dict):
                        return x
                    if isinstance(x, str):
                        try:
                            # 先尝试 json.loads
                            return json.loads(x)
                        except (json.JSONDecodeError, ValueError):
                            try:
                                # 如果是 Python 字典格式的字符串，使用 ast.literal_eval
                                import ast
                                return ast.literal_eval(x)
                            except (ValueError, SyntaxError):
                                logger.warning(f"无法解析 similarity_scores: {str(x)[:100]}...")
                                return {}
                    else:
                        return {}
                
                self.df['similarity_scores'] = self.df['similarity_scores'].apply(safe_parse_similarity_scores)
            else:
                logger.warning("CSV文件中没有找到 'similarity_scores' 字段，创建空字段")
                self.df['similarity_scores'] = [{}] * len(self.df)
                
        elif results_data:
            self.df = pd.DataFrame(results_data)
        else:
            raise ValueError("Either results_csv_path or results_data must be provided")
    
    def analyze_retrieval_patterns(self) -> Dict[str, Any]:
        """
        分析retrieval模式
        
        Returns:
            分析结果字典
        """
        analysis = {}
        
        # 统计每层的retrieval情况
        layer_stats = {}
        total_questions = len(self.df)
        questions_with_retrieval = 0
        
        for idx, row in self.df.iterrows():
            retrieval_records = row.get('retrieval_records', {})
            if retrieval_records and isinstance(retrieval_records, dict) and retrieval_records:
                questions_with_retrieval += 1
                for layer_name, layer_data in retrieval_records.items():
                    if layer_name not in layer_stats:
                        layer_stats[layer_name] = {
                            'total_questions': 0,
                            'avg_retrieved_frames': 0,
                            'timestamp_ranges': []
                        }
                    
                    layer_stats[layer_name]['total_questions'] += 1
                    
                    # 统计retrieved frames数量
                    if 'retrieved_indices' in layer_data and layer_data['retrieved_indices']:
                        indices = layer_data['retrieved_indices']
                        if isinstance(indices, list) and len(indices) > 0:
                            if isinstance(indices[0], list):
                                # 多batch情况
                                avg_frames = np.mean([len(batch) for batch in indices])
                            else:
                                # 单batch情况
                                avg_frames = len(indices)
                            layer_stats[layer_name]['avg_retrieved_frames'] += avg_frames
                    
                    # 统计时间戳范围
                    if 'timestamps' in layer_data and layer_data['timestamps']:
                        timestamps = layer_data['timestamps']
                        if isinstance(timestamps, list) and len(timestamps) > 0:
                            if isinstance(timestamps[0], list):
                                # 多batch情况
                                for batch_ts in timestamps:
                                    if batch_ts:
                                        layer_stats[layer_name]['timestamp_ranges'].append(
                                            (min(batch_ts), max(batch_ts))
                                        )
                            else:
                                # 单batch情况
                                if timestamps:
                                    layer_stats[layer_name]['timestamp_ranges'].append(
                                        (min(timestamps), max(timestamps))
                                    )
        
        # 计算平均值
        for layer_name in layer_stats:
            if layer_stats[layer_name]['total_questions'] > 0:
                layer_stats[layer_name]['avg_retrieved_frames'] /= layer_stats[layer_name]['total_questions']
        
        analysis['layer_statistics'] = layer_stats
        analysis['total_questions'] = total_questions
        analysis['questions_with_retrieval'] = questions_with_retrieval
        
        # 分析retrieval时间分布
        analysis['temporal_distribution'] = self._analyze_temporal_distribution()
        
        # 分析相似度分数
        analysis['similarity_analysis'] = self._analyze_similarity_scores()
        
        return analysis
    
    def _analyze_temporal_distribution(self) -> Dict[str, Any]:
        """分析时间分布"""
        temporal_stats = {}
        
        for idx, row in self.df.iterrows():
            retrieval_records = row.get('retrieval_records', {})
            if retrieval_records and isinstance(retrieval_records, dict):
                for layer_name, layer_data in retrieval_records.items():
                    if layer_name not in temporal_stats:
                        temporal_stats[layer_name] = []
                    
                    if 'timestamps' in layer_data and layer_data['timestamps']:
                        timestamps = layer_data['timestamps']
                        if isinstance(timestamps, list) and len(timestamps) > 0:
                            if isinstance(timestamps[0], list):
                                # 多batch情况
                                for batch_ts in timestamps:
                                    if batch_ts:
                                        temporal_stats[layer_name].extend(batch_ts)
                            else:
                                # 单batch情况
                                temporal_stats[layer_name].extend(timestamps)
        
        # 计算统计量
        distribution_stats = {}
        for layer_name, timestamps in temporal_stats.items():
            if timestamps:
                distribution_stats[layer_name] = {
                    'mean': np.mean(timestamps),
                    'std': np.std(timestamps),
                    'min': np.min(timestamps),
                    'max': np.max(timestamps),
                    'median': np.median(timestamps),
                    'count': len(timestamps)
                }
        
        return distribution_stats
    
    def _analyze_similarity_scores(self) -> Dict[str, Any]:
        """分析相似度分数"""
        similarity_stats = {}
        
        for idx, row in self.df.iterrows():
            similarity_scores = row.get('similarity_scores', {})
            if similarity_scores and isinstance(similarity_scores, dict):
                for layer_name, scores in similarity_scores.items():
                    if layer_name not in similarity_stats:
                        similarity_stats[layer_name] = []
                    
                    if scores and isinstance(scores, list):
                        if len(scores) > 0 and isinstance(scores[0], list):
                            # 多batch情况
                            for batch_scores in scores:
                                if batch_scores:
                                    similarity_stats[layer_name].extend(batch_scores)
                        else:
                            # 单batch情况
                            similarity_stats[layer_name].extend(scores)
        
        # 计算统计量
        score_stats = {}
        for layer_name, scores in similarity_stats.items():
            if scores:
                score_stats[layer_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'median': np.median(scores),
                    'count': len(scores)
                }
        
        return score_stats
    
    def plot_retrieval_distribution(self, save_path: str = None):
        """
        绘制retrieval分布图
        
        Args:
            save_path: 保存路径
        """
        analysis = self.analyze_retrieval_patterns()
        
        # 检查是否有可用的数据
        if not analysis['layer_statistics']:
            print("警告: 没有找到有效的retrieval数据，无法生成图表")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Retrieval Analysis ({analysis["questions_with_retrieval"]}/{analysis["total_questions"]} questions with retrieval)', fontsize=16)
        
        # 1. 每层平均retrieved frames数量
        layer_names = list(analysis['layer_statistics'].keys())
        if layer_names:
            avg_frames = [analysis['layer_statistics'][layer]['avg_retrieved_frames'] 
                         for layer in layer_names]
            
            axes[0, 0].bar(layer_names, avg_frames)
            axes[0, 0].set_title('Average Retrieved Frames per Layer')
            axes[0, 0].set_xlabel('Layer')
            axes[0, 0].set_ylabel('Average Frames')
            axes[0, 0].tick_params(axis='x', rotation=45)
        else:
            axes[0, 0].text(0.5, 0.5, 'No retrieval data available', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Average Retrieved Frames per Layer')
        
        # 2. 时间分布
        if analysis['temporal_distribution']:
            layer_name = list(analysis['temporal_distribution'].keys())[0]  # 取第一层作为示例
            timestamps = []
            for idx, row in self.df.iterrows():
                retrieval_records = row.get('retrieval_records', {})
                if retrieval_records and isinstance(retrieval_records, dict):
                    if layer_name in retrieval_records:
                        layer_data = retrieval_records[layer_name]
                        if 'timestamps' in layer_data and layer_data['timestamps']:
                            ts = layer_data['timestamps']
                            if isinstance(ts, list) and len(ts) > 0:
                                if isinstance(ts[0], list):
                                    timestamps.extend([t for batch in ts for t in batch if batch])
                                else:
                                    timestamps.extend(ts)
            
            if timestamps:
                axes[0, 1].hist(timestamps, bins=min(20, len(set(timestamps))), alpha=0.7)
                axes[0, 1].set_title(f'Timestamp Distribution ({layer_name})')
                axes[0, 1].set_xlabel('Timestamp (seconds)')
                axes[0, 1].set_ylabel('Frequency')
            else:
                axes[0, 1].text(0.5, 0.5, 'No timestamp data available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Timestamp Distribution')
        else:
            axes[0, 1].text(0.5, 0.5, 'No temporal data available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Timestamp Distribution')
        
        # 3. 相似度分数分布
        if analysis['similarity_analysis']:
            layer_name = list(analysis['similarity_analysis'].keys())[0]  # 取第一层作为示例
            scores = []
            for idx, row in self.df.iterrows():
                similarity_scores = row.get('similarity_scores', {})
                if similarity_scores and isinstance(similarity_scores, dict):
                    if layer_name in similarity_scores:
                        score_data = similarity_scores[layer_name]
                        if score_data and isinstance(score_data, list):
                            if len(score_data) > 0 and isinstance(score_data[0], list):
                                scores.extend([s for batch in score_data for s in batch if batch])
                            else:
                                scores.extend(score_data)
            
            if scores:
                axes[1, 0].hist(scores, bins=min(20, len(set(scores))), alpha=0.7)
                axes[1, 0].set_title(f'Similarity Score Distribution ({layer_name})')
                axes[1, 0].set_xlabel('Similarity Score')
                axes[1, 0].set_ylabel('Frequency')
            else:
                axes[1, 0].text(0.5, 0.5, 'No similarity data available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Similarity Score Distribution')
        else:
            axes[1, 0].text(0.5, 0.5, 'No similarity data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Similarity Score Distribution')
        
        # 4. 层间比较
        if len(layer_names) > 1 and analysis['temporal_distribution']:
            layer_means = []
            valid_layers = []
            for layer in layer_names:
                if layer in analysis['temporal_distribution']:
                    layer_means.append(analysis['temporal_distribution'][layer]['mean'])
                    valid_layers.append(layer)
            
            if layer_means:
                axes[1, 1].bar(valid_layers, layer_means)
                axes[1, 1].set_title('Mean Retrieval Timestamp by Layer')
                axes[1, 1].set_xlabel('Layer')
                axes[1, 1].set_ylabel('Mean Timestamp (seconds)')
                axes[1, 1].tick_params(axis='x', rotation=45)
            else:
                axes[1, 1].text(0.5, 0.5, 'No layer comparison data available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Mean Retrieval Timestamp by Layer')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient layers for comparison', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Mean Retrieval Timestamp by Layer')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def export_retrieval_summary(self, save_path: str):
        """
        导出retrieval摘要
        
        Args:
            save_path: 保存路径
        """
        analysis = self.analyze_retrieval_patterns()
        
        # 创建摘要报告
        summary = {
            'total_questions': len(self.df),
            'layer_statistics': analysis['layer_statistics'],
            'temporal_distribution': analysis['temporal_distribution'],
            'similarity_analysis': analysis['similarity_analysis']
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Retrieval summary exported to {save_path}")
    
    def get_retrieval_timeline(self, video_id: str = None, question_idx: int = None) -> Dict[str, Any]:
        """
        获取特定视频或问题的retrieval时间线
        
        Args:
            video_id: 视频ID
            question_idx: 问题索引
            
        Returns:
            时间线数据
        """
        if video_id:
            filtered_df = self.df[self.df['video_id'] == video_id]
        elif question_idx is not None:
            filtered_df = self.df.iloc[question_idx:question_idx+1]
        else:
            filtered_df = self.df.iloc[:1]  # 默认取第一个
        
        timeline_data = {}
        
        for idx, row in filtered_df.iterrows():
            if 'retrieval_records' in row and row['retrieval_records']:
                timeline_data[f"question_{idx}"] = {
                    'video_id': row.get('video_id', 'unknown'),
                    'question': row.get('question', 'unknown'),
                    'retrieval_records': row['retrieval_records'],
                    'similarity_scores': row.get('similarity_scores', {})
                }
        
        return timeline_data


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze retrieval frame records')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to results CSV file')
    parser.add_argument('--output_dir', type=str, default='./analysis_output', help='Output directory')
    parser.add_argument('--video_id', type=str, help='Specific video ID to analyze')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = RetrievalAnalyzer(results_csv_path=args.csv_path)
    
    # 执行分析
    analysis_results = analyzer.analyze_retrieval_patterns()
    print("Analysis completed!")
    
    # 生成可视化
    
    import os
    args.output_dir = os.path.dirname(args.csv_path)
    os.makedirs(args.output_dir, exist_ok=True)
    
    plot_path = os.path.join(args.output_dir, 'retrieval_analysis.png')
    analyzer.plot_retrieval_distribution(save_path=plot_path)
    
    # 导出摘要
    summary_path = os.path.join(args.output_dir, 'retrieval_summary.json')
    analyzer.export_retrieval_summary(summary_path)
    
    # 如果指定了video_id，生成时间线
    if args.video_id:
        timeline = analyzer.get_retrieval_timeline(video_id=args.video_id)
        timeline_path = os.path.join(args.output_dir, f'timeline_{args.video_id}.json')
        with open(timeline_path, 'w', encoding='utf-8') as f:
            json.dump(timeline, f, indent=2, ensure_ascii=False)
        print(f"Timeline saved to {timeline_path}") 