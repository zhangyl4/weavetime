export VIDEO_MIN_PIXELS=78400 # 100*28*28. the minimum visual frame tokens sent to llm is 100
export FPS_MAX_FRAMES=768 # maximum number of frames for each video (480/60/2 = 4min)
export VIDEO_MAX_PIXELS=4816896 # 4816896


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK NODE_RANK
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

learning_rate=1e-5
run_name="livecc_sft_timestamp_shuffle_6clips_llava_lr$learning_rate"


mkdir -p outputs/$run_name
TOKENIZERS_PARALLELISM=false torchrun --nnodes=1 --nproc_per_node=8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
   train_qwen_vl_lora.py \
  --deepspeed /2024233235/videollm-online/configs/deepspeed/zero1.json \
  --output_dir outputs/$run_name \
  --overwrite_output_dir True \
  --run_name $run_name \
  --save_on_each_node True \
  --do_train True \
  --eval_strategy no \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate $learning_rate \
  --warmup_ratio 0.03 \
  --optim adamw_torch \
  --lr_scheduler_type cosine  \
  --num_train_epochs 1 \
  --logging_steps 5 \
  --save_steps 100 \
  --bf16 True \
  --tf32 True \
  --gradient_checkpointing True \
  --pretrained_model_name_or_path chenjoya/LiveCC-7B-Instruct \
  --annotation_paths \
    /2024233235/videollm-online/EyeWO2/data/llava_video_178k_with_seeks_sample_valid.jsonl \
  --dataloader_num_workers 8 \
  --freeze_modules visual \
  --use_liger_kernel True \
  --report_to tensorboard \
  --enable_segment_reconstruction True \
  --segment_reconstruction_paths \
    /2024233235/videollm-online/EyeWO2/data/llava_video_178k_with_seeks_sample_valid.jsonl \
  --segment_reconstruction_clip_number 6 \
  2>&1 | tee outputs/$run_name/train.log
  # --resume_from_checkpoint outputs/$run_name/checkpoint-300 \
  # --eval_steps 200


