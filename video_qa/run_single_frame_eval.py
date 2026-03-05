#!/usr/bin/env python3
"""
单帧错误案例评估运行器
基于ReKV框架的run_eval.py结构，专门用于单帧错误案例分析
"""

import os
import argparse
import subprocess
import multiprocessing
import json
from pathlib import Path


def exec(cmd, sub=False, device=None):
    print(f'exec: {cmd}')
    if not sub:
        if isinstance(cmd, list):
            cmd = ' '.join(cmd)
        os.system(cmd)
    else:
        my_env = os.environ.copy()
        if device is not None:
            gpu_list = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
            my_env["CUDA_VISIBLE_DEVICES"] = str(gpu_list[int(device)])
        subprocess.run(cmd, env=my_env)


def eval_single_frame_error_cases(args):
    """
    单帧错误案例评估
    
    Args:
        args: 命令行参数
    """
    num_chunks = args.num_chunks
    
    # 解析案例文件路径与名称
    case_file = args.case_file if getattr(args, 'case_file', None) else f"/root/videollm-online/ReKV/error_cases/{args.case_type}.json"
    case_stem = Path(case_file).stem

    # 构建保存目录：包含JSON名与关键超参
    save_dir = f"results/{args.model}/single_frame_error/{case_stem}_{args.strategy}_r{args.retrieve_size}-c{args.chunk_size}-fps{args.sample_fps}"
    
    # 选择求解器
    solver = "rekv_single_frame_vqa"
    
    # 案例文件路径已在上方解析
    
    print(f"开始单帧错误案例评估")
    print(f"案例类型: {args.case_type}")
    print(f"选择策略: {args.strategy}")
    print(f"案例文件: {case_file}")
    print(f"结果目录: {save_dir}")
    
    # 检查错误案例文件是否存在
    if not os.path.exists(case_file):
        print(f"错误: 案例文件 {case_file} 不存在")
        print("请先运行 generate_error_case_files.py 生成错误案例文件")
        return
    
    # 创建结果目录
    os.makedirs(save_dir, exist_ok=True)
    
    if not args.only_eval:
        # QA处理
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--case_file", case_file,
                    "--strategy", args.strategy,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--retrieve_chunk_size", str(args.chunk_size),
                    "--save_dir", save_dir,
                    "--anno_path", args.anno_path,
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            
            # 添加可选参数
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
        
        # 合并结果
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    
    # 评估结果
    print(f"开始评估单帧错误案例结果...")
    exec(f"python video_qa/eval/eval_single_frame_error.py --save_dir {save_dir} --case_type {args.case_type} --strategy {args.strategy}")
    
    print(f"单帧错误案例评估完成，结果保存在: {save_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Single Frame Error Case Evaluation')
    parser.add_argument("--model", type=str, default="llava_ov_7b", 
                       choices=['llava_ov_0.5b', 'llava_ov_7b', 'llava_ov_72b', 'video_llava_7b', 'longva_7b', 'qwen2vl_7b'])
    parser.add_argument("--model_path", type=str, default=None, 
                       help="自定义模型权重路径")
    parser.add_argument("--case_type", type=str, required=True,
                       choices=['no_temporal_overlap', 'high_temporal_recall', 'low_temporal_recall'],
                       help="错误案例类型")
    parser.add_argument("--case_file", type=str, default=None, help="错误案例JSON文件路径，优先于case_type")
    parser.add_argument("--strategy", type=str, default="first",
                       choices=['first', 'middle', 'last', 'random'],
                       help="帧选择策略")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--only_eval", action="store_true", help="只进行评估，不重新运行QA")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=1, help="检索帧数，单帧分析固定为1")
    parser.add_argument("--chunk_size", type=int, default=1, help="块大小，单帧分析固定为1")
    parser.add_argument("--debug", type=str, default='false')
    parser.add_argument("--layer_weight_path", type=str, default=None)
    parser.add_argument("--head_weight_path", type=str, default=None)
    parser.add_argument("--anno_path", type=str, default="data/qaego4d/test_mc.json", help="标注文件路径")
    
    args = parser.parse_args()
    
    # 强制设置单帧参数
    args.retrieve_size = 1
    args.chunk_size = 1
    
    print(f'执行单帧错误案例评估: {args.case_type} with {args.strategy} strategy')
    eval_single_frame_error_cases(args)


if __name__ == "__main__":
    main()
