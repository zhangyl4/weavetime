import torch
import numpy as np
from logzero import logger

from video_qa.base import BaseVQA, StreamVideoEncodingMixin, work
from video_qa.mixins import TimePromptMixin, StandardRetrievalQAMixin


class ReKVStreamTimePromptVQA(BaseVQA, StreamVideoEncodingMixin, TimePromptMixin, StandardRetrievalQAMixin):
    def __init__(self, anno, save_dir, sample_fps,
                 qa_model, qa_processor=None,
                 num_chunks=None, chunk_idx=None,
                 retrieve_size=64, chunk_size=1, query_type='question', use_dynamic_size=False,
                 input_fps=None,**kwargs):
        """
        Args:
            input_fps: Input frequency for feeding frames (Hz). Must be <= sample_fps.
                      If None, defaults to sample_fps.
        """
        super().__init__(anno, save_dir, sample_fps, qa_model, qa_processor,
                        num_chunks, chunk_idx, retrieve_size, chunk_size, query_type, use_dynamic_size, **kwargs)
        
        # Input fps for grouping frames (must be <= sample_fps)
        self.input_fps = input_fps if input_fps is not None else sample_fps
        assert self.input_fps <= self.sample_fps, \
            f'input_fps ({self.input_fps}) must be <= sample_fps ({self.sample_fps})'
        
        # Calculate how many frames per input
        self.frames_per_input = int(round(self.sample_fps / self.input_fps))
        
        # Modify processor's chat_template to preserve content order
        self._setup_chat_template()
    


if __name__ == "__main__":
    work(ReKVStreamTimePromptVQA)
