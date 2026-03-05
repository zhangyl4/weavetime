export CUDA_VISIBLE_DEVICES=0,1,2,4,5
num_chunks=5

# Supported model: llava_ov_0.5b llava_ov_7b llava_ov_72b video_llava_7b longva_7b qwen2vl_7b
model=qwen2vl_7b

# Supported dataset: qaego4d egoschema cgbench mlvu activitynet_qa rvs_ego rvs_movie eventhal vidhalluc_ach_mcq
# MLVU has an extremely long video (~9hr). Remove it in the annotation file if your system doesn't have enough RAM.
dataset=ovobench_layered # streamingbench_layered
export PYTHONPATH=/2024233235/videollm-online/ReKV:$PYTHONPATH
python -m video_qa.run_eval \
    --num_chunks $num_chunks \
    --model ${model} \
    --dataset ${dataset} \
    --sample_fps 1 \
    --input_fps 1 \
    --n_local 15680 \
    --retrieve_size 128 \
    --use_hybrid_similarity true \
    --convert_to_streaming true \
    --use_dynamic_size false \
    --query_type prompt \
    --short_memory_layers 0 \
    --model_path /root/videollm-online/livecc/outputs/qwen2vl_sft_timestamp_shuffle_llava_lr1e-5/ \
    --time_prompt \


export CUDA_VISIBLE_DEVICES=0,1,5,6

num_chunks=4

model=llava_ov_7b

dataset=ovobench_layered
export PYTHONPATH=/root/videollm-online/ReKV:$PYTHONPATH
python -m video_qa.run_eval \
    --num_chunks $num_chunks \
    --model ${model} \
    --dataset ${dataset} \
    --sample_fps 1 \
    --input_fps 1 \
    --n_local 15680 \
    --retrieve_size 64 \
    --short_memory_layers 0 \
    --time_prompt \
    --use_hybrid_similarity true \
    --use_dynamic_size false \
    --query_type prompt \
    --model_path /root/videollm-online/livecc/outputs/llavaov_sft_timestamp_next_et_cof_lr1e-5/ \
