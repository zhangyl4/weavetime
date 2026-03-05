import os
import argparse
import subprocess
import multiprocessing
import json
import random
import math


def exec(cmd, sub=False, device=None):
    print(f'exec: {cmd}')
    if not sub:
        if isinstance(cmd, list):
            cmd = ' '.join(cmd)
        os.system(cmd)
    else:
        my_env = os.environ.copy()
        gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        my_env["CUDA_VISIBLE_DEVICES"] = str(gpu_list[int(device)])
        subprocess.run(cmd, env=my_env)


def extend_common_flags(cmd, args):
    """Append common storage/filter flags to the subprocess command if provided."""
    # storage
    if getattr(args, 'storage_mode', None):
        cmd.extend(["--storage_mode", str(args.storage_mode)])
    if getattr(args, 'storage_save_rate', None) is not None:
        cmd.extend(["--storage_save_rate", str(args.storage_save_rate)])
    if getattr(args, 'storage_sim_threshold', None) is not None:
        cmd.extend(["--storage_sim_threshold", str(args.storage_sim_threshold)])
    # frame filter
    if getattr(args, 'frame_filter_mode', None):
        cmd.extend(["--frame_filter_mode", str(args.frame_filter_mode)])
    if getattr(args, 'frame_filter_rate', None) is not None:
        cmd.extend(["--frame_filter_rate", str(args.frame_filter_rate)])
    if getattr(args, 'frame_filter_sim_threshold', None) is not None:
        cmd.extend(["--frame_filter_sim_threshold", str(args.frame_filter_sim_threshold)])
    # query type
    if getattr(args, 'query_type', None):
        cmd.extend(["--query_type", str(args.query_type)])
    # dynamic size
    if getattr(args, 'use_dynamic_size', None):
        cmd.extend(["--use_dynamic_size", str(args.use_dynamic_size)])
    # merge-load-kv
    if getattr(args, 'merge_load_kv', None):
        cmd.extend(["--merge_load_kv", str(args.merge_load_kv)])
    # convert to streaming
    if getattr(args, 'convert_to_streaming', None) is not None:
        cmd.extend(["--convert_to_streaming", str(args.convert_to_streaming)])
    if getattr(args, 'time_prompt', False):
        cmd.append("--time_prompt")
    # prompt builder type
    if getattr(args, 'prompt_builder_type', None):
        cmd.extend(["--prompt_builder_type", str(args.prompt_builder_type)])
    if getattr(args, 'entropy_threshold', None) is not None:
        cmd.extend(["--entropy_threshold", str(args.entropy_threshold)])
    if getattr(args, 'entropy_window_layers', None) is not None:
        cmd.extend(["--entropy_window_layers", str(args.entropy_window_layers)])
    # precomputed retrieval file
    if getattr(args, 'precomputed_retrieval_file', None):
        cmd.extend(["--precomputed_retrieval_file", str(args.precomputed_retrieval_file)])
    return cmd


def eval_mlvu(args):
    num_chunks = args.num_chunks
    time_tag = "time_prompt_" if getattr(args, 'time_prompt', False) else ""
    base_tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
    tag = f"{time_tag}{base_tag}"
    save_dir = f"results/{args.model}/mlvu/{tag}{args.retrieve_size}-{args.sample_fps}"
    if getattr(args, 'baseline', 'false') == 'true':
        solver = "baseline_offline_vqa"
    else:
        solver = "rekv_offline_time_prompt_vqa" if getattr(args, 'time_prompt', False) else "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/mlvu/dev_debug_mc_filtered.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            # forward similarity flag
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            # Add input_fps for time prompt mode
            if getattr(args, 'time_prompt', False) and getattr(args, 'input_fps', None):
                cmd.extend(["--input_fps", str(args.input_fps)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_mlvu_layered(args):
    num_chunks = args.num_chunks
    # 构建分层描述符
    time_tag = "time_prompt_" if getattr(args, 'time_prompt', False) else ""
    suffix_parts = []
    if getattr(args, 'entropy_threshold', None) is not None:
        suffix_parts.append(f"et{str(args.entropy_threshold).replace('.', 'p')}")
    if getattr(args, 'entropy_window_layers', None) is not None:
        suffix_parts.append(f"ew{args.entropy_window_layers}")
    entropy_suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""

    if args.short_memory_layers:
        layer_desc = "_".join(map(str, args.short_memory_layers))
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/mlvu/layered_{layer_desc}_{time_tag}{tag}{args.retrieve_size}-{args.sample_fps}{entropy_suffix}"
    else:
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/mlvu/uniform_{time_tag}{tag}{args.retrieve_size}-{args.sample_fps}{entropy_suffix}"
    
    if getattr(args, 'baseline', 'false') == 'true':
        solver = "baseline_offline_vqa"
    else:
        solver = "rekv_offline_time_prompt_recent_vqa" if getattr(args, 'time_prompt', False) else "rekv_offline_vqa_recent"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/mlvu/dev_debug_mc_filtered.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            
            if args.short_memory_layers:
                cmd.extend(["--short_memory_layers"] + list(map(str, args.short_memory_layers)))
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            # forward similarity flag
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            # Add input_fps for time prompt mode
            if getattr(args, 'time_prompt', False) and getattr(args, 'input_fps', None):
                cmd.extend(["--input_fps", str(args.input_fps)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_qaego4d(args):
    num_chunks = args.num_chunks
    tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
    
    # Support time prompt mode
    if getattr(args, 'time_prompt', False):
        tag = "time_prompt_" + tag
        solver = "rekv_offline_time_prompt_vqa" if getattr(args, 'stream_mode', False) == False else "rekv_stream_time_prompt_vqa"
    else:
        solver = "baseline_offline_vqa" if getattr(args, 'baseline', 'false') == 'true' else "rekv_offline_vqa"
    
    save_dir = f"results/{args.model}/qaego4d/{tag}{args.retrieve_size}-{args.sample_fps}"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/qaego4d/test_mc.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            # Add input_fps for time prompt mode
            if getattr(args, 'time_prompt', False) and getattr(args, 'input_fps', None):
                cmd.extend(["--input_fps", str(args.input_fps)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_qaego4d_temporal_shift(args):
    """
    Temporal window relocation experiment on QAEgo4D:
    - Sample 100 unique videos (1 question each)
    - For each of five positions (head, 25%, 50%, 75%, tail), shift the retrieval window (annotation-only)
    - Run evaluation with a specialized solver that constrains retrieval after load
    - Save per-position results and a mapping CSV of moved positions
    """
    random_seed = 2025
    rng = random.Random(random_seed)

    # Prepare save directories
    time_tag = "time_prompt_" if getattr(args, 'time_prompt', False) else ""
    base_tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
    tag = f"{time_tag}{base_tag}"
    base_save_dir = f"results/{args.model}/qaego4d_temporal_shift/{tag}{args.retrieve_size}-{args.sample_fps}"
    exec(f"mkdir -p {base_save_dir}")
    manifests_dir = f"{base_save_dir}/manifests"
    exec(f"mkdir -p {manifests_dir}")

    # Load dataset
    anno_path = "data/qaego4d/test_mc.json"
    with open(anno_path, "r") as f:
        ds = json.load(f)

    # Deterministically select 100 unique videos, one question each
    videos = list(ds)
    rng.shuffle(videos)
    selected = []
    for v in videos:
        if 'conversations' not in v or len(v['conversations']) == 0:
            continue
        sample = rng.choice(v['conversations'])
        # Ensure a valid temporal window exists
        tw = sample.get('temporal_windows', None)
        if not tw or not isinstance(tw, list) or len(tw) == 0 or len(tw[0]) != 2:
            continue
        selected.append({
            'video_id': v['video_id'],
            'video_path': v['video_path'],
            'duration': v.get('duration', None),
            'conversations': [sample]
        })
        if len(selected) >= 100:
            break

    if len(selected) == 0:
        print("No valid samples with temporal_windows found in dataset.")
        return

    # Helper to compute shifted window while preserving length
    def shift_window(orig_start, orig_end, duration, label):
        length = max(0.0, float(orig_end) - float(orig_start))
        D = float(duration) if duration is not None else max(1.0, float(orig_end))
        def clamp_start(s):
            return max(0.0, min(s, max(0.0, D - length)))
        if label == 'head':
            start = 0.0
        elif label == 'q25':
            start = clamp_start(0.25 * D - 0.5 * length)
        elif label == 'mid':
            start = clamp_start(0.50 * D - 0.5 * length)
        elif label == 'q75':
            start = clamp_start(0.75 * D - 0.5 * length)
        elif label == 'tail':
            start = clamp_start(D - length)
        else:
            start = orig_start
        end = start + length
        # Round for stability
        return round(start, 3), round(end, 3)

    positions = [
        ('head', 'head'),
        ('q25', 'q25'),
        ('mid', 'mid'),
        ('q75', 'q75'),
        ('tail', 'tail')
    ]

    # Write position-specific manifests and the moved_positions.csv
    moved_rows = []
    manifest_paths = {}
    for key, label in positions:
        manifest = []
        for item in selected:
            conv = item['conversations'][0]
            orig_tw = conv.get('temporal_windows', [[0.0, 0.0]])[0]
            o_start, o_end = float(orig_tw[0]), float(orig_tw[1])
            s, e = shift_window(o_start, o_end, item.get('duration', None), key)
            # build a copy with audit fields (do not overwrite original GT temporal_windows)
            new_conv = dict(conv)
            new_conv['moved_to'] = {'label': label, 'start': s, 'end': e}
            new_conv['orig_window'] = {'start': round(o_start, 3), 'end': round(o_end, 3)}
            new_item = {
                'video_id': item['video_id'],
                'video_path': item['video_path'],
                'duration': item.get('duration', None),
                'conversations': [new_conv]
            }
            manifest.append(new_item)
            moved_rows.append({
                'video_id': item['video_id'],
                'question': conv.get('question', ''),
                'label': label,
                'orig_start': round(o_start, 3),
                'orig_end': round(o_end, 3),
                'new_start': s,
                'new_end': e
            })
        manifest_path = f"{manifests_dir}/qaego4d_shift_{label}.json"
        with open(manifest_path, "w") as mf:
            json.dump(manifest, mf, ensure_ascii=False, indent=2)
        manifest_paths[label] = manifest_path

    # Write moved_positions.csv
    moved_csv = f"{base_save_dir}/moved_positions.csv"
    if moved_rows:
        # simple CSV writer
        import csv
        with open(moved_csv, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(moved_rows[0].keys()))
            writer.writeheader()
            for r in moved_rows:
                writer.writerow(r)

    # Launch evaluations, one per position, using the temporal-shift solver
    results_dirs = {}
    for _, label in positions:
        save_dir = f"{base_save_dir}/{label}"
        exec(f"mkdir -p {save_dir}")
        if not args.only_eval:
            processes = []
            for idx in range(0, args.num_chunks):
                cmd = ["python", "video_qa/rekv_offline_vqa_temporal_shift.py",
                       "--model", args.model,
                       "--sample_fps", str(args.sample_fps),
                       "--n_local", str(args.n_local),
                       "--retrieve_size", str(args.retrieve_size),
                       "--save_dir", save_dir,
                       "--anno_path", manifest_paths[label],
                       "--debug", args.debug,
                       "--num_chunks", str(args.num_chunks),
                       "--chunk_idx", str(idx)]
                if getattr(args, 'model_path', None):
                    cmd.extend(["--model_path", args.model_path])
                if getattr(args, 'layer_weight_path', None):
                    cmd.extend(["--layer_weight_path", args.layer_weight_path])
                if getattr(args, 'head_weight_path', None):
                    cmd.extend(["--head_weight_path", args.head_weight_path])
                # forward similarity flag
                if hasattr(args, 'use_hybrid_similarity'):
                    cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
                # Add input_fps for time prompt mode
                if getattr(args, 'time_prompt', False) and getattr(args, 'input_fps', None):
                    cmd.extend(["--input_fps", str(args.input_fps)])
                cmd = extend_common_flags(cmd, args)
                # llava_ov_72b multi-gpu env var handling
                gpu_spec = f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)
                p = multiprocessing.Process(target=exec, args=(cmd, True, gpu_spec))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
            # merge per-chunk results
            exec(f"> {save_dir}/results.csv")
            for idx in range(args.num_chunks):
                if idx == 0:
                    exec(f"head -n 1 {save_dir}/{args.num_chunks}_{idx}.csv > {save_dir}/results.csv")
                exec(f"tail -n +2 {save_dir}/{args.num_chunks}_{idx}.csv >> {save_dir}/results.csv")
                exec(f"rm {save_dir}/{args.num_chunks}_{idx}.csv")
        # eval per label
        exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")
        results_dirs[label] = save_dir

    # Optionally, summarize across positions (simple copy of per-label metrics paths)
    summary_json = {
        'positions': {label: {'results_dir': rd} for label, rd in results_dirs.items()},
        'moved_positions_csv': moved_csv
    }
    with open(f"{base_save_dir}/summary.json", "w") as sf:
        json.dump(summary_json, sf, ensure_ascii=False, indent=2)

def eval_qaego4d_layered(args):
    """QAEgo4D数据集分层检索实验"""
    num_chunks = args.num_chunks
    # 构建分层描述符
    time_tag = "time_prompt_" if getattr(args, 'time_prompt', False) else ""
    suffix_parts = []
    if getattr(args, 'entropy_threshold', None) is not None:
        suffix_parts.append(f"et{str(args.entropy_threshold).replace('.', 'p')}")
    if getattr(args, 'entropy_window_layers', None) is not None:
        suffix_parts.append(f"ew{args.entropy_window_layers}")
    entropy_suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""

    if args.short_memory_layers:
        layer_desc = "_".join(map(str, args.short_memory_layers))
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/qaego4d/layered_{layer_desc}_{time_tag}{tag}{args.retrieve_size}-{args.sample_fps}{entropy_suffix}"
    else:
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/qaego4d/uniform_{time_tag}{tag}{args.retrieve_size}-{args.sample_fps}{entropy_suffix}"
    
    if getattr(args, 'baseline', 'false') == 'true':
        solver = "baseline_offline_vqa"
    else:
        solver = "rekv_offline_time_prompt_recent_vqa" if getattr(args, 'time_prompt', False) else "rekv_offline_vqa_recent"
    
    print(f"开始QAEgo4D分层检索实验，短期记忆层: {args.short_memory_layers}")
    
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/qaego4d/test_mc.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            
            # 添加短期记忆层参数
            if args.short_memory_layers:
                cmd.extend(["--short_memory_layers"] + list(map(str, args.short_memory_layers)))
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            # Add input_fps for time prompt mode
            if getattr(args, 'time_prompt', False) and getattr(args, 'input_fps', None):
                cmd.extend(["--input_fps", str(args.input_fps)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")
    print(f"分层检索实验完成，结果保存在: {save_dir}")

def eval_egoschema(args):
    num_chunks = args.num_chunks
    time_tag = "time_prompt_" if getattr(args, 'time_prompt', False) else ""
    base_tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
    tag = f"{time_tag}{base_tag}"
    save_dir = f"results/{args.model}/egoschema/{tag}{args.retrieve_size}-{args.sample_fps}"
    if getattr(args, 'baseline', 'false') == 'true':
        solver = "baseline_offline_vqa"
    else:
        solver = "rekv_offline_time_prompt_vqa" if getattr(args, 'time_prompt', False) else "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/egoschema/full.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            if getattr(args, 'time_prompt', False) and getattr(args, 'input_fps', None):
                cmd.extend(["--input_fps", str(args.input_fps)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_egoschema.py --save_dir {save_dir}")

def eval_egoschema_layered(args):
    """EgoSchema数据集分层检索实验"""
    num_chunks = args.num_chunks
    
    # 构建分层描述符
    is_head_weight = args.head_weight_path is not None
    is_layer_weight = args.layer_weight_path is not None
    if args.short_memory_layers:
        layer_desc = "_".join(map(str, args.short_memory_layers))
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/egoschema/layered_{layer_desc}_{tag}{args.retrieve_size}-{args.sample_fps}_{is_head_weight}_{is_layer_weight}"
    else:
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/egoschema/uniform_{tag}{args.retrieve_size}-{args.sample_fps}_{is_head_weight}_{is_layer_weight}"
    
    solver = "baseline_offline_vqa" if getattr(args, 'baseline', 'false') == 'true' else "rekv_offline_vqa_recent"
    
    print(f"开始EgoSchema分层检索实验，短期记忆层: {args.short_memory_layers}")
    
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/egoschema/full.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            
            # 添加短期记忆层参数
            if args.short_memory_layers:
                cmd.extend(["--short_memory_layers"] + list(map(str, args.short_memory_layers)))
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_egoschema.py --save_dir {save_dir}")
    print(f"分层检索实验完成，结果保存在: {save_dir}")

def eval_activitynet_qa(args):
    num_chunks = args.num_chunks
    tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
    save_dir = f"results/{args.model}/activitynet_qa/{tag}{args.retrieve_size}-{args.sample_fps}"
    solver = "baseline_offline_vqa" if getattr(args, 'baseline', 'false') == 'true' else "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/activitynet_qa/test.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")

def eval_activitynet_qa_layered(args):
    """ActivityNet-QA数据集分层检索实验"""
    num_chunks = args.num_chunks
    
    # 构建分层描述符
    is_head_weight = args.head_weight_path is not None
    is_layer_weight = args.layer_weight_path is not None
    if args.short_memory_layers:
        layer_desc = "_".join(map(str, args.short_memory_layers))
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/activitynet_qa/layered_{layer_desc}_{tag}{args.retrieve_size}-{args.sample_fps}_{is_head_weight}_{is_layer_weight}"
    else:
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/activitynet_qa/uniform_{tag}{args.retrieve_size}-{args.sample_fps}_{is_head_weight}_{is_layer_weight}"
    
    solver = "baseline_offline_vqa" if getattr(args, 'baseline', 'false') == 'true' else "rekv_offline_vqa_recent"
    
    print(f"开始ActivityNet-QA分层检索实验，短期记忆层: {args.short_memory_layers}")
    
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/activitynet_qa/test.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            
            # 添加短期记忆层参数
            if args.short_memory_layers:
                cmd.extend(["--short_memory_layers"] + list(map(str, args.short_memory_layers)))
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")
    print(f"分层检索实验完成，结果保存在: {save_dir}")



def eval_rvs_ego(args):
    num_chunks = args.num_chunks
    time_tag = "time_prompt_" if getattr(args, 'time_prompt', False) else ""
    base_tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
    save_dir = f"results/{args.model}/rvs_ego/{time_tag}{base_tag}{args.retrieve_size}-{args.sample_fps}"
    if getattr(args, 'baseline', 'false') == 'true':
        solver = "baseline_stream_vqa"
    else:
        solver = "rekv_stream_time_prompt_vqa" if getattr(args, 'time_prompt', False) else "rekv_stream_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/rvs/ego/ego4d_oe.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            # Add input_fps for time prompt mode
            if getattr(args, 'time_prompt', False) and getattr(args, 'input_fps', None):
                cmd.extend(["--input_fps", str(args.input_fps)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")

def eval_rvs_movie(args):
    num_chunks = args.num_chunks
    time_tag = "time_prompt_" if getattr(args, 'time_prompt', False) else ""
    base_tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
    save_dir = f"results/{args.model}/rvs_movie/{time_tag}{base_tag}{args.retrieve_size}-{args.sample_fps}"
    if getattr(args, 'baseline', 'false') == 'true':
        solver = "baseline_stream_vqa"
    else:
        solver = "rekv_stream_time_prompt_vqa" if getattr(args, 'time_prompt', False) else "rekv_stream_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/rvs/movie/movienet_oe.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            if getattr(args, 'time_prompt', False) and getattr(args, 'input_fps', None):
                cmd.extend(["--input_fps", str(args.input_fps)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")

def eval_rvs_ego_layered(args):
    """RVS Ego数据集分层检索实验（流式）"""
    num_chunks = args.num_chunks
    
    # 构建分层描述符
    is_head_weight = args.head_weight_path is not None
    is_layer_weight = args.layer_weight_path is not None
    if args.short_memory_layers:
        layer_desc = "_".join(map(str, args.short_memory_layers))
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/rvs_ego/layered_{layer_desc}_{tag}{args.retrieve_size}-{args.sample_fps}_{is_head_weight}_{is_layer_weight}"
    else:
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/rvs_ego/uniform_{tag}{args.retrieve_size}-{args.sample_fps}_{is_head_weight}_{is_layer_weight}"
    
    solver = "baseline_stream_vqa" if getattr(args, 'baseline', 'false') == 'true' else "rekv_stream_vqa_recent"
    
    print(f"开始RVS Ego分层检索实验，短期记忆层: {args.short_memory_layers}")
    
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/rvs/ego/ego4d_oe.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            
            # 添加短期记忆层参数
            if args.short_memory_layers:
                cmd.extend(["--short_memory_layers"] + list(map(str, args.short_memory_layers)))
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")
    print(f"分层检索实验完成，结果保存在: {save_dir}")

def eval_rvs_movie_layered(args):
    """RVS Movie数据集分层检索实验（流式）"""
    num_chunks = args.num_chunks
    
    # 构建分层描述符
    is_head_weight = args.head_weight_path is not None
    is_layer_weight = args.layer_weight_path is not None
    if args.short_memory_layers:
        layer_desc = "_".join(map(str, args.short_memory_layers))
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/rvs_movie/layered_{layer_desc}_{tag}{args.retrieve_size}-{args.sample_fps}_{is_head_weight}_{is_layer_weight}"
    else:
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/rvs_movie/uniform_{tag}{args.retrieve_size}-{args.sample_fps}_{is_head_weight}_{is_layer_weight}"
    
    solver = "baseline_stream_vqa" if getattr(args, 'baseline', 'false') == 'true' else "rekv_stream_vqa_recent"
    
    print(f"开始RVS Movie分层检索实验，短期记忆层: {args.short_memory_layers}")
    
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/rvs/movie/movienet_oe.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            
            # 添加短期记忆层参数
            if args.short_memory_layers:
                cmd.extend(["--short_memory_layers"] + list(map(str, args.short_memory_layers)))
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")
    print(f"分层检索实验完成，结果保存在: {save_dir}")

def eval_ovobench(args):
    num_chunks = args.num_chunks
    time_tag = "time_prompt_" if getattr(args, 'time_prompt', False) else ""
    base_tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
    save_dir = f"results/{args.model}/ovobench/{time_tag}{base_tag}{args.retrieve_size}-{args.sample_fps}"
    if getattr(args, 'baseline', 'false') == 'true':
        solver = "baseline_stream_vqa"
    else:
        solver = "rekv_stream_time_prompt_vqa" if getattr(args, 'time_prompt', False) else "rekv_stream_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/ovobench/ovo_bench_rekv.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            if getattr(args, 'time_prompt', False) and getattr(args, 'input_fps', None):
                cmd.extend(["--input_fps", str(args.input_fps)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_ovobench.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp")

def eval_ovobench_layered(args):
    """OVOBench数据集分层检索实验（流式）"""
    num_chunks = args.num_chunks
    
    # 构建分层描述符
    time_tag = "time_prompt_" if getattr(args, 'time_prompt', False) else ""
    suffix_parts = []
    if getattr(args, 'entropy_threshold', None) is not None:
        suffix_parts.append(f"et{str(args.entropy_threshold).replace('.', 'p')}")
    if getattr(args, 'entropy_window_layers', None) is not None:
        suffix_parts.append(f"ew{args.entropy_window_layers}")
    entropy_suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""

    if args.short_memory_layers:
        layer_desc = "_".join(map(str, args.short_memory_layers))
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/ovobench/layered_{layer_desc}_{time_tag}{tag}{args.retrieve_size}-{args.sample_fps}{entropy_suffix}"
    else:
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/ovobench/uniform_{time_tag}{tag}{args.retrieve_size}-{args.sample_fps}{entropy_suffix}"
    
    if getattr(args, 'baseline', 'false') == 'true':
        solver = "baseline_stream_vqa"
    else:
        solver = "rekv_stream_time_prompt_recent_vqa" if getattr(args, 'time_prompt', False) else "rekv_stream_vqa_recent"
    
    print(f"开始OVOBench分层检索实验，短期记忆层: {args.short_memory_layers}")
    
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/ovobench/ovo_bench_rekv.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            
            # 添加短期记忆层参数
            if args.short_memory_layers:
                cmd.extend(["--short_memory_layers"] + list(map(str, args.short_memory_layers)))
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            if getattr(args, 'time_prompt', False) and getattr(args, 'input_fps', None):
                cmd.extend(["--input_fps", str(args.input_fps)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_ovobench.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp")
    print(f"分层检索实验完成，结果保存在: {save_dir}")

def eval_etbench(args):
    num_chunks = args.num_chunks
    tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
    save_dir = f"results/{args.model}/etbench/{tag}{args.retrieve_size}-{args.sample_fps}"
    # Default to stream solver as ETBench is video-centric; adjust if needed later
    solver = "baseline_offline_vqa" if getattr(args, 'baseline', 'false') == 'true' else "rekv_offline_vqa"
    if not args.only_eval:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/etbench/etbench_rekv.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval via official ETBench evaluator wrapper
    hf_root = "/root/.cache/huggingface/hub/datasets--PolyU-ChenLab--ETBench/snapshots/2d7bce92b69624c3c26ac054e5d7947463568283"
    exec(f"python video_qa/eval/eval_etbench.py --save_dir {save_dir} --hf_root {hf_root}")



def eval_etbench_layered(args):
    """ETBench 数据集分层检索实验"""
    num_chunks = args.num_chunks
    
    # 构建分层描述符
    is_head_weight = args.head_weight_path is not None
    is_layer_weight = args.layer_weight_path is not None
    if args.short_memory_layers:
        layer_desc = "_".join(map(str, args.short_memory_layers))
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/etbench/layered_{layer_desc}_{tag}{args.retrieve_size}-{args.sample_fps}_{is_head_weight}_{is_layer_weight}"
    else:
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/etbench/uniform_{tag}{args.retrieve_size}-{args.sample_fps}_{is_head_weight}_{is_layer_weight}"
    
    solver = "baseline_offline_vqa" if getattr(args, 'baseline', 'false') == 'true' else "rekv_offline_vqa_recent"
    
    print(f"开始ETBench分层检索实验，短期记忆层: {args.short_memory_layers}")
    
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/etbench/etbench_rekv.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            
            # 添加短期记忆层参数
            if args.short_memory_layers:
                cmd.extend(["--short_memory_layers"] + list(map(str, args.short_memory_layers)))
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval via official ETBench evaluator wrapper
    hf_root = "/root/.cache/huggingface/hub/datasets--PolyU-ChenLab--ETBench/snapshots/2d7bce92b69624c3c26ac054e5d7947463568283"
    exec(f"python video_qa/eval/eval_etbench.py --save_dir {save_dir} --hf_root {hf_root}")

def eval_streamingbench(args):
    num_chunks = args.num_chunks
    tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
    save_dir = f"results/{args.model}/streamingbench/{tag}{args.retrieve_size}-{args.sample_fps}"
    solver = "baseline_stream_vqa" if getattr(args, 'baseline', 'false') == 'true' else "rekv_stream_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/StreamingBench/streamingbench_rekv.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
            
    # eval
    exec(f"python video_qa/eval/eval_streamingbench.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp")


def eval_videomme(args):
    num_chunks = args.num_chunks
    time_tag = "time_prompt_" if getattr(args, 'time_prompt', False) else ""
    base_tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
    tag = f"{time_tag}{base_tag}"
    save_dir = f"results/{args.model}/videomme/{tag}{args.retrieve_size}-{args.sample_fps}"
    if getattr(args, 'baseline', 'false') == 'true':
        solver = "baseline_offline_vqa"
    else:
        solver = "rekv_offline_time_prompt_vqa" if getattr(args, 'time_prompt', False) else "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/videomme/videomme_rekv.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            # forward similarity flag
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            # Add input_fps for time prompt mode
            if getattr(args, 'time_prompt', False) and getattr(args, 'input_fps', None):
                cmd.extend(["--input_fps", str(args.input_fps)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_streamingbench.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp")

def eval_streamingbench_layered(args):
    """StreamingBench数据集分层检索实验（流式）"""
    time_tag = "time_prompt_" if getattr(args, 'time_prompt', False) else ""
    base_tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
    tag = f"{time_tag}{base_tag}"
    num_chunks = args.num_chunks
    
    # 构建分层描述符
    is_head_weight = args.head_weight_path is not None
    is_layer_weight = args.layer_weight_path is not None
    if args.short_memory_layers:
        layer_desc = "_".join(map(str, args.short_memory_layers))
        save_dir = f"results/{args.model}/streamingbench/layered_{layer_desc}_{time_tag}{base_tag}{args.retrieve_size}-{args.sample_fps}_{is_head_weight}_{is_layer_weight}"
    else:
        save_dir = f"results/{args.model}/streamingbench/uniform_{time_tag}{base_tag}{args.retrieve_size}-{args.sample_fps}_{is_head_weight}_{is_layer_weight}"
    
    if getattr(args, 'baseline', 'false') == 'true':
        solver = "baseline_stream_vqa"
    else:
        solver = "rekv_stream_time_prompt_recent_vqa" if getattr(args, 'time_prompt', False) else "rekv_stream_vqa_recent"
    
    print(f"开始StreamingBench分层检索实验，短期记忆层: {args.short_memory_layers}")
    
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/StreamingBench/streamingbench_rekv.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            
            # 添加短期记忆层参数
            if args.short_memory_layers:
                cmd.extend(["--short_memory_layers"] + list(map(str, args.short_memory_layers)))
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_streamingbench.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp")
    print(f"分层检索实验完成，结果保存在: {save_dir}")


def eval_cgbench(args):
    num_chunks = args.num_chunks
    tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
    save_dir = f"results/{args.model}/cgbench/{tag}{args.retrieve_size}-{args.sample_fps}"
    if getattr(args, 'baseline', 'false') == 'true':
        solver = "baseline_offline_vqa"
    else:
        solver = "rekv_offline_time_prompt_vqa" if getattr(args, 'time_prompt', False) else "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/cgbench/full_mc.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")


def eval_eventhal(args):
    num_chunks = args.num_chunks
    
    time_tag = "time_prompt_" if getattr(args, 'time_prompt', False) else ""
    tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
    save_dir = f"results/{args.model}/eventhal/{tag}{args.retrieve_size}-{args.sample_fps}"
    if getattr(args, 'baseline', 'false') == 'true':
        solver = "baseline_offline_vqa"
    else:
        solver = "rekv_offline_time_prompt_vqa" if getattr(args, 'time_prompt', False) else "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/eventhal/all_questions.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_binary.py --save_dir {save_dir}")


def eval_eventhal_layered(args):
    """EventHal 数据集分层检索实验 (offline)."""
    num_chunks = args.num_chunks
    # 构建分层描述符
    time_tag = "time_prompt_" if getattr(args, 'time_prompt', False) else ""
    suffix_parts = []
    if getattr(args, 'entropy_threshold', None) is not None:
        suffix_parts.append(f"et{str(args.entropy_threshold).replace('.', 'p')}")
    if getattr(args, 'entropy_window_layers', None) is not None:
        suffix_parts.append(f"ew{args.entropy_window_layers}")
    entropy_suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""

    if args.short_memory_layers:
        layer_desc = "_".join(map(str, args.short_memory_layers))
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/eventhal/layered_{layer_desc}_{time_tag}{tag}{args.retrieve_size}-{args.sample_fps}{entropy_suffix}"
    else:
        tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
        save_dir = f"results/{args.model}/eventhal/uniform_{time_tag}{tag}{args.retrieve_size}-{args.sample_fps}{entropy_suffix}"

    if getattr(args, 'baseline', 'false') == 'true':
        solver = "baseline_offline_vqa"
    else:
        solver = "rekv_offline_time_prompt_recent_vqa" if getattr(args, 'time_prompt', False) else "rekv_offline_vqa_recent"

    print(f"开始EventHal分层检索实验，短期记忆层: {args.short_memory_layers}")

    if not args.only_eval:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/eventhal/all_questions.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]

            if args.short_memory_layers:
                cmd.extend(["--short_memory_layers"] + list(map(str, args.short_memory_layers)))
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            if getattr(args, 'time_prompt', False) and getattr(args, 'input_fps', None):
                cmd.extend(["--input_fps", str(args.input_fps)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_binary.py --save_dir {save_dir}")


def eval_vidhalluc_ach_mcq(args):
    num_chunks = args.num_chunks
    tag = "baseline_" if getattr(args, 'baseline', 'false') == 'true' else ""
    save_dir = f"results/{args.model}/vidhalluc_ach_mcq/{tag}{args.retrieve_size}-{args.sample_fps}"
    solver = "baseline_offline_vqa" if getattr(args, 'baseline', 'false') == 'true' else "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/vidhalluc/ach_mcq_rekv.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--chunk_idx", str(idx)]
            if getattr(args, 'model_path', None):
                cmd.extend(["--model_path", args.model_path])
            if getattr(args, 'layer_weight_path', None):
                cmd.extend(["--layer_weight_path", args.layer_weight_path])
            if getattr(args, 'head_weight_path', None):
                cmd.extend(["--head_weight_path", args.head_weight_path])
            if hasattr(args, 'use_hybrid_similarity'):
                cmd.extend(["--use_hybrid_similarity", str(args.use_hybrid_similarity)])
            cmd = extend_common_flags(cmd, args)
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava_ov_7b", choices=['llava_ov_0.5b', 'llava_ov_7b', 'llava_ov_72b', 'video_llava_7b', 'longva_7b', 'qwen2vl_7b'])
    parser.add_argument("--model_path", type=str, default=None, 
                        help="自定义模型权重路径，如果提供则覆盖默认路径")
    parser.add_argument("--dataset", type=str, default=None, choices=['mlvu', 'mlvu_layered',
                                                                      'qaego4d', 'qaego4d_recent', 'qaego4d_layered', 'qaego4d_temporal_shift',
                                                                      'egoschema', 'egoschema_layered', 'activitynet_qa', 'activitynet_qa_layered', 
                                                                      'rvs_ego', 'rvs_ego_layered', 'rvs_movie', 'rvs_movie_layered', 
                                                                      'cgbench', 'eventhal', 'eventhal_layered', 'vidhalluc_ach_mcq', 'ovobench', 'ovobench_layered', 'etbench', 'videomme',
                                                                      'etbench_layered',
                                                                      'streamingbench', 'streamingbench_layered'])
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--debug", type=str, default='false')
    parser.add_argument("--short_memory_layers", type=int, nargs='*', default=None, 
                        help="使用短期记忆（最近帧）的层索引列表，例如: --short_memory_layers 0 1 2")
    parser.add_argument("--layer_weight_path", type=str, default=None)
    parser.add_argument("--head_weight_path", type=str, default=None)
    parser.add_argument("--use_hybrid_similarity", type=str, default='false', help="是否启用混合similarity (true/false)")
    parser.add_argument("--query_type", type=str, default='question', choices=['question', 'prompt'], 
                        help="指定查询类型：question或prompt")
    parser.add_argument("--use_dynamic_size", type=str, default='false', choices=['true', 'false'], 
                        help="是否启用动态检索大小 (true/false)")
    parser.add_argument("--merge_load_kv", type=str, default='false', choices=['true', 'false'],
                        help="是否启用合并式KV加载 (true/false)")
    parser.add_argument("--baseline", type=str, default='false', choices=['true', 'false'],
                        help="当为true时，使用baseline_offline_vqa，仅采样retrieve_size帧作答")
    # Time prompt mode args
    parser.add_argument("--time_prompt", action="store_true",
                        help="启用时间戳提示模式")
    parser.add_argument("--input_fps", type=float, default=None,
                        help="时间戳提示的输入频率 (必须 <= sample_fps)")
    parser.add_argument("--stream_mode", action="store_true",
                        help="使用stream模式（与time_prompt一起使用）")
    # storage & frame filter args (pass-through to subprocess)
    parser.add_argument("--storage_mode", type=str, default=None)
    parser.add_argument("--storage_save_rate", type=float, default=None)
    parser.add_argument("--storage_sim_threshold", type=float, default=None)
    parser.add_argument("--frame_filter_mode", type=str, default=None)
    parser.add_argument("--frame_filter_rate", type=float, default=None)
    parser.add_argument("--frame_filter_sim_threshold", type=float, default=None)
    parser.add_argument("--convert_to_streaming", type=str, default='false', choices=['true', 'false', 'baseline'],
                        help="是否转换为流式模型 (true/false)")
    parser.add_argument("--prompt_builder_type", type=str, default=None, choices=['streamingbench', 'instruction_rich', 'compact', 'numbered'])
    parser.add_argument("--entropy_threshold", type=float, default=0.6,
                        help="覆盖熵自适应检索阈值 (默认 0.6)")
    parser.add_argument("--entropy_window_layers", type=int, default=1,
                        help="熵平均时使用的最近层数 (默认 1)")
    parser.add_argument("--precomputed_retrieval_file", type=str, default=None,
                        help="预计算检索结果的CSV文件路径，如果提供则使用其中的retrieval indices避免重复检索")
    args = parser.parse_args()
    func_dic = {
        'mlvu': eval_mlvu,
        'mlvu_layered': eval_mlvu_layered,
        'qaego4d': eval_qaego4d,
        'qaego4d_temporal_shift': eval_qaego4d_temporal_shift,
        'qaego4d_layered': eval_qaego4d_layered,
        'egoschema': eval_egoschema,
        'egoschema_layered': eval_egoschema_layered,
        'activitynet_qa': eval_activitynet_qa,
        'activitynet_qa_layered': eval_activitynet_qa_layered,
        'rvs_ego': eval_rvs_ego,
        'rvs_ego_layered': eval_rvs_ego_layered,
        'rvs_movie': eval_rvs_movie,
        'rvs_movie_layered': eval_rvs_movie_layered,
        'cgbench': eval_cgbench,
        'eventhal': eval_eventhal,
        'eventhal_layered': eval_eventhal_layered,
        'vidhalluc_ach_mcq': eval_vidhalluc_ach_mcq,
        'ovobench': eval_ovobench,
        'ovobench_layered': eval_ovobench_layered,
        'etbench': eval_etbench,
        'etbench_layered': eval_etbench_layered,
        'streamingbench': eval_streamingbench,
        'streamingbench_layered': eval_streamingbench_layered,
        'videomme': eval_videomme,
    }
    if args.dataset in func_dic:
        print(f'Execute {args.dataset} evaluation')
        func_dic[args.dataset](args)
