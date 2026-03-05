# The number of processes utilized for parallel evaluation.
# Normally, set it to the number of GPUs on your machine.
# Yet, llava_ov_72b needs 4x 80GB GPUs. So set num_chunks to num_gpus//4.

export CUDA_VISIBLE_DEVICES=1,2,3,4,5
num_chunks=5

# Supported model: llava_ov_0.5b llava_ov_7b llava_ov_72b video_llava_7b longva_7b qwen2vl_7b
model=llava_ov_7b

# Supported dataset: qaego4d egoschema cgbench mlvu activitynet_qa rvs_ego rvs_movie eventhal vidhalluc_ach_mcq
# MLVU has an extremely long video (~9hr). Remove it in the annotation file if your system doesn't have enough RAM.
dataset=ovobench

export PYTHONPATH=/root/videollm-online/ReKV:$PYTHONPATH

# llavaov in this codebase
python -m video_qa.run_eval \
    --num_chunks $num_chunks \
    --model ${model} \
    --dataset ${dataset} \
    --sample_fps 1 \
    --n_local 15000 \
    --retrieve_size 64 \
    --use_hybrid_similarity false \
    --baseline true \
    --convert_to_streaming baseline \

# llavaov + rekv in this codebase
python -m video_qa.run_eval \
    --num_chunks $num_chunks \
    --model ${model} \
    --dataset ${dataset} \
    --sample_fps 1 \
    --input_fps 1 \
    --n_local 15000 \
    --retrieve_size 64 \
    --use_hybrid_similarity false \
    --convert_to_streaming false \
    --use_dynamic_size false \
    --query_type question \

# n_local : window size
# retrieve_size : retrieve frame number
# chunk_size : chunk frame number

# retrieve_size*frame_token_number < n_local
# chunk_size*frame_token_number < n_local