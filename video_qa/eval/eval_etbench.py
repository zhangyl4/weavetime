import os
import json
import argparse
import glob
import shutil
from typing import Dict, Any, List, Tuple

import pandas as pd


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(text.strip().lower().split())


def build_hf_gt_index(hf_root: str, modality: str = "vid") -> Tuple[Dict[Tuple[str, str, int], Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]], List[Dict[str, Any]]]:
    ann_dir = os.path.join(hf_root, "annotations", modality)
    key_by_triplet: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    key_by_video_q: Dict[Tuple[str, str], Dict[str, Any]] = {}
    key_by_basename_q: Dict[Tuple[str, str], Dict[str, Any]] = {}

    json_paths = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    for jp in json_paths:
        try:
            with open(jp, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for sample in data:
            task = sample.get("task")
            source = sample.get("source")
            idx = sample.get("idx")
            if task is None or source is None or idx is None:
                continue
            video = sample.get("video", "")
            q = normalize_text(sample.get("q", ""))
            key_by_triplet[(task, source, int(idx))] = sample
            key_by_video_q[(video, q)] = sample
            key_by_basename_q[(os.path.basename(video), q)] = sample

    # Also return a flat list for fallback scans
    all_samples: List[Dict[str, Any]] = list(key_by_triplet.values())
    return key_by_triplet, key_by_video_q, key_by_basename_q, all_samples


def pick_pred_text(row: Dict[str, Any]) -> str:
    if "a" in row and isinstance(row["a"], str):
        return row["a"]
    if "pred_answer" in row and isinstance(row["pred_answer"], str):
        return row["pred_answer"]
    if "pred" in row and isinstance(row["pred"], str):
        return row["pred"]
    if "prediction" in row and isinstance(row["prediction"], str):
        return row["prediction"]
    return ""


def normalize_video_id(v: str) -> str:
    if v is None:
        return ""
    # Remove file extension and get basename
    basename = os.path.basename(str(v))
    return os.path.splitext(basename)[0]


def load_predictions_csv(pred_path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(pred_path)
    # Do NOT hard-filter by retrieve_size/chunk_size; keep all rows.
    # Some runs may use different settings, and downstream matching is agnostic to these.
    return df.to_dict(orient="records")


def enrich_samples_with_gt(pred_rows: List[Dict[str, Any]],
                           key_by_triplet: Dict[Tuple[str, str, int], Dict[str, Any]],
                           key_by_video_q: Dict[Tuple[str, str], Dict[str, Any]],
                           key_by_basename_q: Dict[Tuple[str, str], Dict[str, Any]],
                           all_gt_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    used_triplets: set = set()
    for row in pred_rows:
        # Determine prediction text
        pred_text = pick_pred_text(row)

        # Try matching order: explicit triplet -> (video, q) -> (basename, q)
        matched = None
        task = row.get("task")
        source = row.get("source")
        idx = row.get("idx")
        if task is not None and source is not None and idx is not None:
            key = (str(task), str(source), int(idx))
            if key not in used_triplets:
                matched = key_by_triplet.get(key)

        if matched is None:
            video_id = row.get("video") or row.get("video_id") or row.get("video_path") or row.get("vid")
            q = row.get("question") or row.get("q") or ""
            qn = normalize_text(str(q))
            if video_id is not None:
                key = (str(video_id), qn)
                matched = key_by_video_q.get(key)
            if matched is None:
                key2 = (normalize_video_id(str(video_id)), qn)
                matched = key_by_basename_q.get(key2)
            
            # Try flexible question matching: check if GT question is a prefix of pred question
            if matched is None and video_id is not None:
                vbase = normalize_video_id(str(video_id))
                for gt in all_gt_list:
                    if normalize_video_id(gt.get("video", "")) == vbase:
                        gt_q = normalize_text(gt.get("q", ""))
                        # Check if GT question is a prefix of prediction question
                        if gt_q and qn.startswith(gt_q):
                            t = gt.get("task"); s = gt.get("source"); i = gt.get("idx")
                            key = (str(t), str(s), int(i))
                            if key not in used_triplets:
                                matched = gt
                                break

        # Fallback 1: match by basename only (first unused)
        if matched is None and video_id is not None:
            vbase = normalize_video_id(str(video_id))
            for gt in all_gt_list:
                if normalize_video_id(gt.get("video", "")) == vbase:
                    t = gt.get("task"); s = gt.get("source"); i = gt.get("idx")
                    key = (str(t), str(s), int(i))
                    if key not in used_triplets:
                        matched = gt
                        break

        if matched is None:
            # Last resort: cannot find GT; skip
            continue

        sample: Dict[str, Any] = {
            "task": matched.get("task"),
            "source": matched.get("source"),
            "idx": matched.get("idx"),
            "a": pred_text,
        }

        # task-specific fields copied from GT
        for key in ("o", "p", "tgt", "g", "video", "duration"):
            if key in matched:
                sample[key] = matched[key]

        out.append(sample)
        used_triplets.add((str(sample["task"]), str(sample["source"]), int(sample["idx"])))
    return out


def save_jsonl(samples: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def call_official_evaluator(hf_root: str, jsonl_path: str, work_dir: str) -> None:
    evaluator = os.path.join(hf_root, "evaluation", "compute_metrics.py")
    cmd = f"python {evaluator} {jsonl_path}"
    os.system(cmd)
    # Move outputs if generated alongside jsonl
    metrics_json = os.path.join(os.path.dirname(jsonl_path), "metrics.json")
    metrics_log = os.path.join(os.path.dirname(jsonl_path), "metrics.log")
    dst_json = os.path.join(work_dir, "metrics.json")
    dst_log = os.path.join(work_dir, "metrics.log")
    if os.path.exists(metrics_json) and os.path.abspath(metrics_json) != os.path.abspath(dst_json):
        shutil.copy(metrics_json, dst_json)
    if os.path.exists(metrics_log) and os.path.abspath(metrics_log) != os.path.abspath(dst_log):
        shutil.copy(metrics_log, dst_log)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="Directory containing results.csv; outputs also saved here")
    parser.add_argument("--pred_path", type=str, default=None, help="Optional direct path to predictions CSV")
    parser.add_argument("--hf_root", type=str, default="/root/.cache/huggingface/hub/datasets--PolyU-ChenLab--ETBench/snapshots/2d7bce92b69624c3c26ac054e5d7947463568283")
    parser.add_argument("--modality", type=str, default="vid", choices=["vid", "txt"])
    return parser.parse_args()


def main():
    args = parse_args()
    pred_csv = args.pred_path if args.pred_path else os.path.join(args.save_dir, "results.csv")
    if not os.path.exists(pred_csv):
        print(f"[W] predictions not found: {pred_csv}")
    else:
        print(f"[I] load predictions: {pred_csv}")

    print(f"[I] load HF ETBench annotations from: {args.hf_root}")
    triplet_idx, video_q_idx, basename_q_idx, all_gt_list = build_hf_gt_index(args.hf_root, args.modality)

    pred_rows = load_predictions_csv(pred_csv) if os.path.exists(pred_csv) else []
    samples = enrich_samples_with_gt(pred_rows, triplet_idx, video_q_idx, basename_q_idx, all_gt_list)

    jsonl_path = os.path.join(args.save_dir, "etbench_pred.jsonl")
    save_jsonl(samples, jsonl_path)
    print(f"[I] wrote {len(samples)} samples to {jsonl_path}")

    # If no samples matched, skip evaluator to avoid upstream crashes
    if len(samples) == 0:
        print("[W] No samples matched GT. Skip official evaluator.")
        return
    call_official_evaluator(args.hf_root, jsonl_path, args.save_dir)
    print(f"[I] evaluation finished. See metrics.json / metrics.log in {args.save_dir}")


if __name__ == "__main__":
    main()


