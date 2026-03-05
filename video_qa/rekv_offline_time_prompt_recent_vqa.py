import torch
import numpy as np
from logzero import logger

from video_qa.base import BaseVQA, OfflineVideoEncodingMixin, work
from video_qa.mixins import (
    TimePromptMixin,
    EntropyAdaptiveRecentRetrievalMixin,
    StandardRetrievalQAMixin,
    WeightLoadingMixin
)


class ReKVOfflineTimePromptVQARecent(BaseVQA, OfflineVideoEncodingMixin, TimePromptMixin,
                                      EntropyAdaptiveRecentRetrievalMixin, StandardRetrievalQAMixin, WeightLoadingMixin):
    def __init__(
        self,
        anno,
        save_dir,
        sample_fps,
        qa_model,
        qa_processor=None,
        num_chunks=None,
        chunk_idx=None,
        retrieve_size=64,
        chunk_size=1,
        query_type="question",
        use_dynamic_size=False,
        input_fps=None,
        short_memory_layers=None,
        layer_weight_path=None,
        head_weight_path=None,
        merge_load_kv=False,
        **kwargs,
    ):
        super().__init__(
            anno,
            save_dir,
            sample_fps,
            qa_model,
            qa_processor,
            num_chunks,
            chunk_idx,
            retrieve_size,
            chunk_size,
            query_type,
            use_dynamic_size,
            **kwargs,
        )
        
        # Time-prompt input fps control
        self.input_fps = input_fps if input_fps is not None else sample_fps
        assert self.input_fps <= self.sample_fps, \
            f'input_fps ({self.input_fps}) must be <= sample_fps ({self.sample_fps})'
        self.frames_per_input = int(round(self.sample_fps / self.input_fps))

        # Recent-layered retrieval options
        self.short_memory_layers = short_memory_layers or []
        logger.info(f"Short memory layers: {self.short_memory_layers}")
        self.merge_load_kv = bool(merge_load_kv)

        # Load optional weights
        self._load_layer_and_head_weights(layer_weight_path, head_weight_path)

        # Ensure chat template preserves content order for time prompts
        self._setup_chat_template()


if __name__ == "__main__":
    work(ReKVOfflineTimePromptVQARecent)
