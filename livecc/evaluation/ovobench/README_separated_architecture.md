# OVOBench Evaluation with Separated Forward and Generate Architecture

这个文档说明了如何使用新的分离式架构进行OVOBench评估，该架构将forward pass和generation分离，并提供了灵活的past_key_values处理接口。

## 架构概述

新的架构将原来的单一generate过程分为两个步骤：

1. **Forward Pass**: 使用MRopeSinkCache进行forward pass，获取past_key_values
2. **Generation**: 使用获取的past_key_values和answer_prefix进行生成

## 核心函数

### 1. `forward_with_mrope_cache()`

```python
def forward_with_mrope_cache(model, processor, conversation, video_input, mrope_section, window_length=2048, num_sink_tokens=1024):
```

**功能**: 执行forward pass并返回包含缓存key-value对的MRopeSinkCache实例

**参数**:
- `model`: 用于forward pass的模型
- `processor`: 用于tokenization的处理器
- `conversation`: 要处理的对话
- `video_input`: 视频输入
- `mrope_section`: MRoPE配置
- `window_length`: 缓存窗口长度
- `num_sink_tokens`: sink tokens数量

**返回**: MRopeSinkCache实例，包含缓存的key-value对

### 2. `process_past_key_values()`

```python
def process_past_key_values(past_key_values, model, processor, answer_prefix, max_new_tokens=32, custom_processing_func=None):
```

**功能**: 处理past_key_values并使用answer_prefix进行生成

**参数**:
- `past_key_values`: MRopeSinkCache实例
- `model`: 用于生成的模型
- `processor`: 用于tokenization的处理器
- `answer_prefix`: 生成前添加的前缀
- `max_new_tokens`: 最大生成token数
- `custom_processing_func`: 可选的past_key_values自定义处理函数

**返回**: 生成的文本

### 3. `mcq_predict_with_custom_processing()`

```python
def mcq_predict_with_custom_processing(model, processor, benchmark_path, options, remote_loader, custom_processing_func=None, **kwargs):
```

**功能**: 支持自定义past_key_values处理的MCQ预测

**参数**:
- `custom_processing_func`: 处理past_key_values的自定义函数
- `**kwargs`: 其他参数

## 使用示例

### 基本使用

```python
# 1. Forward pass获取past_key_values
past_key_values = forward_with_mrope_cache(
    model=model,
    processor=processor,
    conversation=conversation,
    video_input=video_input,
    mrope_section=mrope_section
)

# 2. 处理past_key_values并生成答案
generated_text = process_past_key_values(
    past_key_values=past_key_values,
    model=model,
    processor=processor,
    answer_prefix='The answer is:\n',
    max_new_tokens=32
)
```

### 自定义处理

```python
def my_custom_processing(past_key_values):
    """自定义past_key_values处理函数"""
    # 在这里可以修改缓存
    # 例如：过滤某些层、应用变换等
    print(f"Processing cache with {len(past_key_values.key_cache)} layers")
    return past_key_values

# 使用自定义处理函数
predictions, results = mcq_predict_with_custom_processing(
    model=model,
    processor=processor,
    benchmark_path=benchmark_path,
    options=options,
    remote_loader=None,
    custom_processing_func=my_custom_processing,
    answer_prefix='The answer is:\n'
)
```

## 架构优势

### 1. 灵活性
- 可以在forward和generate之间插入自定义处理逻辑
- 支持对past_key_values进行修改和优化

### 2. 可调试性
- 可以单独测试forward pass和generation
- 更容易定位性能瓶颈

### 3. 可扩展性
- 支持多种past_key_values处理策略
- 可以轻松添加新的缓存优化技术

### 4. 内存效率
- 更好的内存管理控制
- 可以在处理过程中清理不需要的缓存

## 自定义处理函数示例

### 示例1: 缓存统计

```python
def cache_statistics(past_key_values):
    """统计缓存信息"""
    print(f"Cache layers: {len(past_key_values.key_cache)}")
    print(f"Window length: {past_key_values.window_length}")
    print(f"Sink tokens: {past_key_values.num_sink_tokens}")
    return past_key_values
```

### 示例2: 缓存过滤

```python
def filter_cache_layers(past_key_values, keep_layers=[0, 1, 2]):
    """只保留指定的缓存层"""
    if len(past_key_values.key_cache) > max(keep_layers):
        past_key_values.key_cache = [past_key_values.key_cache[i] for i in keep_layers]
        past_key_values.value_cache = [past_key_values.value_cache[i] for i in keep_layers]
    return past_key_values
```

### 示例3: 缓存压缩

```python
def compress_cache(past_key_values, compression_ratio=0.5):
    """压缩缓存大小"""
    for i in range(len(past_key_values.key_cache)):
        # 示例：保留一半的序列长度
        seq_len = past_key_values.key_cache[i].shape[-2]
        keep_len = int(seq_len * compression_ratio)
        past_key_values.key_cache[i] = past_key_values.key_cache[i][:, :, -keep_len:, :]
        past_key_values.value_cache[i] = past_key_values.value_cache[i][:, :, -keep_len:, :]
    return past_key_values
```

## 性能优化建议

### 1. 缓存配置
```python
# 根据GPU内存调整
window_length = 1024  # 较小的窗口长度
num_sink_tokens = 512  # 较少的sink tokens
```

### 2. 批处理优化
```python
# 使用批处理进行forward pass
batch_conversations = [conv1, conv2, conv3]
batch_videos = [vid1, vid2, vid3]
# 批量处理forward pass
```

### 3. 内存管理
```python
# 及时清理缓存
del past_key_values
torch.cuda.empty_cache()
```

## 故障排除

### 1. 内存不足
- 减少`window_length`和`num_sink_tokens`
- 使用自定义处理函数压缩缓存

### 2. 生成质量下降
- 检查`answer_prefix`格式
- 调整`max_new_tokens`参数

### 3. 性能问题
- 使用自定义处理函数优化缓存
- 考虑批处理forward pass

## 相关文件

- `distributed_evaluate_ovobench_with_mrope.py`: 包含分离式架构的完整实现
- `inference_qwen2vl_memory.py`: 原始MRopeSinkCache实现 