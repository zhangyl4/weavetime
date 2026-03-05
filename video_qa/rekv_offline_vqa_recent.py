import torch
from logzero import logger

from video_qa.base import BaseVQA, OfflineVideoEncodingMixin, work
from video_qa.mixins import (
    EntropyAdaptiveRecentRetrievalMixin,
    StandardRetrievalQAMixin,
    WeightLoadingMixin
)


class ReKVOfflineVQARecent(BaseVQA, OfflineVideoEncodingMixin, EntropyAdaptiveRecentRetrievalMixin,
                            StandardRetrievalQAMixin, WeightLoadingMixin):
    
    def __init__(self, *args, short_memory_layers=None, layer_weight_path=None, head_weight_path=None, merge_load_kv=False, **kwargs):
        """
        初始化ReKV离线VQA模型
        
        Args:
            short_memory_layers: 使用短期记忆（最近帧）的层索引列表，例如 [0, 1, 2]
                               如果为None，则所有层都使用正常检索
            layer_weight_path: 每层权重JSON文件路径
            head_weight_path: 每层每头权重JSON文件路径
        """
        super().__init__(*args, **kwargs)
        self.short_memory_layers = short_memory_layers or []
        logger.info(f"Short memory layers: {self.short_memory_layers}")
        self.merge_load_kv = bool(merge_load_kv)
        
        # Load optional weights
        self._load_layer_and_head_weights(layer_weight_path, head_weight_path)
        

if __name__ == "__main__":
    work(ReKVOfflineVQARecent)
