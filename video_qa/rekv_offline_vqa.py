import torch
from logzero import logger

from video_qa.base import BaseVQA, OfflineVideoEncodingMixin, work
from video_qa.mixins import StandardRetrievalQAMixin


class ReKVOfflineVQA(BaseVQA, OfflineVideoEncodingMixin, StandardRetrievalQAMixin):
    """Offline VQA with standard retrieval."""
    pass





if __name__ == "__main__":
	work(ReKVOfflineVQA)
