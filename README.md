# WeaveTime

The official PyTorch implementation of "WeaveTime: Streaming from Earlier Frames into Emergent Memory in VideoLLMs".

## Overview

WeaveTime is a streaming video question answering system that addresses the memory bottleneck in VideoLLMs by dynamically weaving earlier frame representations into an emergent memory through in-context KV-cache retrieval.


## TODO

- [x] Environment Setup
- [x] Training Code
- [x] Data Preparation
- [x] Model Weights


## Project Structure (core code)

```
WeaveTime/
├── model/                      # Model implementations
│   ├── abstract_rekv.py        # Abstract base class for ReKV
│   ├── attention/              # Attention and KV-cache modules
│   │   └── kv_cache_manager.py
│   ├── llava_onevision_rekv.py
│   ├── qwen2vl_rekv.py
│   ├── longva_rekv.py
│   ├── video_llava_rekv.py
│   └── flash_vstream_rekv.py
├── video_qa/                   # Video QA evaluation
│   ├── base.py                 # Base classes
│   ├── mixin.py                # new function classes
│   ├── rekv_stream_vqa.py      # Streaming VQA
│   ├── rekv_offline_vqa.py     # Offline VQA
│   └── eval/                   # Evaluation scripts
├── livecc/                     # Training code
│   ├── train_llava_ov_lora.py  # LLaVA-OneVision LoRA training
│   ├── train_qwen_vl_lora.py   # Qwen2-VL LoRA training
│   ├── env.sh                  # Training environment setup
│   ├── data/llava_ov_dataset.py# llavaov dataset code
│   ├── data/qwen_vl_dataset.py # qwen2vl dataset code
│   └── scripts/                # Training scripts
├── tools/                      # Analysis tools
├── prepare.sh                  # Environment setup for inference
└── livecc/env.sh               # Environment setup for training
```

## Environment Setup

### For Inference

```bash
# Create conda environment and install dependencies
bash prepare.sh
```

### For Training

```bash
# Create conda environment for training
bash livecc/env.sh
```


## Data Preparation

### 1. Download Original Videos

Follow the instructions from [LLaVA-NeXT](https://github.com/LLaVA-Vi/LLaVA-NeXT) to download the original videos for each benchmark.

### 2. Download JSON Files and Scripts

Download the JSON files and processing scripts from ModelScope:

```bash
modelscope download --dataset zhangyl9/weavetime_it
```

This repository contains:
- JSON annotation files for each benchmark
- some tool script

You must modify them video origin parent path in json.

### 3. Extract Videos

Use the provided script to unzip videos:

```bash
bash unzip_llava.sh
```

## Model Weights

### 1. Download Base Model Weights

#### LLaVA-OneVision (7B)
```bash
# Download from HuggingFace
huggingface-cli download --resume-download llava-hf/llava-onevision-qwen2-7b-ov-hf

```

#### Qwen2-VL (7B)
```bash
# Download from HuggingFace
huggingface-cli download --resume-download Qwen/Qwen2-VL-7B-Instruct

```

### 2. Download WeaveTime LoRA Weights

Download the fine-tuned LoRA weights from ModelScope:

#### LLaVA-OneVision WeaveTime LoRA
```bash
modelscope download --model zhangyl9/llavaov-weavetime
```

#### Qwen2-VL WeaveTime LoRA
```bash
# Coming soon - upload to ModelScope
modelscope download --model zhangyl9/qwen2vl-weavetime
```



## Inference

### Quick Start

```bash
# Run evaluation on video QA benchmarks
# you can refer run_eval.py for more information
bash eval.sh
```

### Supported Models

- LLaVA-OneVision (7B)
- Qwen2-VL (7B)

### Supported Benchmarks

Referring download.sh to download video and soft link them in data/[ benchmark name ]/videos

- StreamingBench
- OVOBench
- MLVU
- QA-EGO4D
- EventHall
- Egoschema

## Training

The `livecc/` directory contains training code for fine-tuning VideoLLMs with WeaveTime.


### Training Scripts
You can refer config in code (train_llava_ov_lora, train_qwen_vl_lora, data/llava_ov_dataset.py, data/qwen_vl_dataset.py) for more details.
```bash
# LLaVA-OneVision LoRA training
bash livecc/scripts/sft_ov_lora_shuffle.sh

# Qwen2-VL LoRA training  
bash livecc/scripts/sft_qwen_vl_lora_shuffle.sh

```

## License

Apache 2.0 License
