#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,5,6
export HF_ENDPOINT=https://hf-mirror.com
export DECORD_EOF_RETRY_MAX=20480
export PYTHONPATH=/root/videollm-online/ReKV:${PYTHONPATH:-}

num_chunks=${NUM_CHUNKS:-4}
model=${MODEL:-llava_ov_7b}
dataset=ovobench_layered
model_path=${MODEL_PATH:-/root/videollm-online/livecc/outputs/llavaov_sft_timestamp_shuffle_et_llava_lr5e-5/}
sample_fps=${SAMPLE_FPS:-1}
input_fps=${INPUT_FPS:-1}
n_local=${N_LOCAL:-15680}
retrieve_size=${RETRIEVE_SIZE:-64}
short_memory_layers=${SHORT_MEMORY_LAYERS:-0}

repo_root=/root/videollm-online/ReKV
results_root=${repo_root}/results/${model}/ovobench
time_tag="time_prompt_"

build_entropy_suffix() {
    local threshold="$1"
    local window="$2"
    local parts=()
    if [[ -n "${threshold}" ]]; then
        parts+=("et${threshold//./p}")
    fi
    if [[ -n "${window}" ]]; then
        parts+=("ew${window}")
    fi
    if (( ${#parts[@]} > 0 )); then
        printf "_%s" "$(IFS=_; echo "${parts[*]}")"
    fi
}

run_cmd() {
    echo "Running: $*"
    eval "$@"
}

common_args=(
    --num_chunks "${num_chunks}"
    --model "${model}"
    --dataset "${dataset}"
    --sample_fps "${sample_fps}"
    --input_fps "${input_fps}"
    --n_local "${n_local}"
    --retrieve_size "${retrieve_size}"
    --use_hybrid_similarity true
    --short_memory_layers "${short_memory_layers}"
    --convert_to_streaming true
    --use_dynamic_size false
    --query_type prompt
    --time_prompt
    --model_path "${model_path}"
)


thresholds=(0.6)
windows=(1)

# thresholds=(0.4)
# windows=(1 2 3 4)

echo "=== Entropy Ablation Grid ==="
for threshold in "${thresholds[@]}"; do
    for window in "${windows[@]}"; do
        echo "--- threshold=${threshold}, window=${window} ---"
        entropy_suffix=$(build_entropy_suffix "${threshold}" "${window}")
        result_dir=${results_root}/layered_${short_memory_layers}_${time_tag}${retrieve_size}-${sample_fps}${entropy_suffix}
        if [[ -d "${result_dir}" ]]; then
            echo "Skip run (exists): ${result_dir}"
            continue
        fi
        run_cmd "python -m video_qa.run_eval ${common_args[*]} --entropy_threshold ${threshold} --entropy_window_layers ${window}"
    done
done
