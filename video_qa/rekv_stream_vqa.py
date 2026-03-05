import torch
import numpy as np
from logzero import logger

from video_qa.base import BaseVQA, StreamVideoEncodingMixin, work


class ReKVStreamVQA(BaseVQA, StreamVideoEncodingMixin):
    """Stream VQA with standard retrieval (no retrieval_info)."""
    pass

if __name__ == "__main__":
    work(ReKVStreamVQA)
