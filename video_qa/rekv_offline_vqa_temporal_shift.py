import warnings
import argparse
from transformers import logging
import logzero
from logzero import logger
from video_qa.base import BaseVQA, OfflineVideoEncodingMixin, work, floor_time_by_fps, ceil_time_by_fps
from video_qa.mixins import StandardRetrievalQAMixin
import pandas as pd
import torch
import math


class ReKVOfflineVQA_TemporalShift(BaseVQA, StandardRetrievalQAMixin):
	"""
	Offline encoding; move the GT clip inside the in-memory video AFTER load and BEFORE encoding.
	Retrieval remains unchanged (model-side), so the clip's new position affects performance.
	"""

	@torch.inference_mode()
	def analyze_a_video(self, video_sample):
		video_path = video_sample['video_path']
		try:
			video = self.load_video(video_path)
		except Exception as e:
			logger.error(f"Error loading video: {e}")
			return
		if not isinstance(video, torch.Tensor):
			video_tensor = torch.from_numpy(video)
		else:
			video_tensor = video

		# Expect 1 conversation per selected video in this experiment
		sample = video_sample['conversations'][0] if video_sample.get('conversations') else None
		# Compute original segment indices (from orig_window or temporal_windows)
		seg_start_s = seg_end_s = None
		if sample is not None:
			if 'orig_window' in sample and isinstance(sample['orig_window'], dict):
				seg_start_s = float(sample['orig_window'].get('start', 0.0))
				seg_end_s = float(sample['orig_window'].get('end', 0.0))
			elif 'temporal_windows' in sample and isinstance(sample['temporal_windows'], list) and len(sample['temporal_windows']) > 0:
				tw = sample['temporal_windows'][0]
				if isinstance(tw, list) and len(tw) == 2:
					seg_start_s = float(tw[0])
					seg_end_s = float(tw[1])

		target_start_s = None
		shift_label = None
		if sample is not None and 'moved_to' in sample and isinstance(sample['moved_to'], dict):
			target_start_s = float(sample['moved_to'].get('start', 0.0))
			shift_label = str(sample['moved_to'].get('label', 'unknown'))

		# If we have a valid original segment and a target position, move the segment within the frame tensor
		if seg_start_s is not None and seg_end_s is not None and seg_end_s > seg_start_s and target_start_s is not None:
			# Convert to frame indices (based on sample_fps and current length)
			N = int(video_tensor.shape[0])
			seg_start = int(max(0, min(int(math.floor(seg_start_s * self.sample_fps)), N - 1)))
			seg_end = int(max(1, min(int(math.ceil(seg_end_s * self.sample_fps)), N)))
			seg_len = max(1, seg_end - seg_start)
			# Target insertion index in original timeline
			target_start = int(max(0, min(int(round(target_start_s * self.sample_fps)), N - seg_len)))

			# Extract segment and remove from timeline
			seg_frames = video_tensor[seg_start:seg_end]
			prefix = video_tensor[:seg_start]
			suffix = video_tensor[seg_end:]
			without_seg = torch.cat([prefix, suffix], dim=0)
			# If inserting after original segment, adjust target index due to removal
			target_index_after_removal = target_start
			if target_start > seg_start:
				target_index_after_removal = max(0, target_start - seg_len)
			# Split and insert
			left = without_seg[:target_index_after_removal]
			right = without_seg[target_index_after_removal:]
			video_tensor = torch.cat([left, seg_frames, right], dim=0)
			# Safety: keep length unchanged
			if int(video_tensor.shape[0]) != N:
				# In unexpected edge case, fallback to original
				logger.warning("Temporal move changed video length unexpectedly; reverting to original order.")
				video_tensor = video if isinstance(video, torch.Tensor) else torch.from_numpy(video)

		# Encode whole video once (offline)
		self.qa_model.clear_cache()
		self.qa_model.encode_init_prompt()

		video_id = video_sample.get('video_id', video_path)
		if hasattr(self, 'encode_video_with_time_prompts'):
			if getattr(self, 'convert_to_streaming', 'false') == 'true':
				encode_chunk_size = getattr(self, 'retrieve_size', 64) // getattr(self, 'retrieve_size', 64)
			else:
				encode_chunk_size = getattr(self, 'retrieve_size', 64)  // 16
			self.encode_video_with_time_prompts(video_tensor, video_id=video_id, encode_chunk_size=encode_chunk_size)
		else:
			if getattr(self, 'convert_to_streaming', 'false') == 'true':
				encode_chunk_size = getattr(self, 'retrieve_size', 64) // getattr(self, 'retrieve_size', 64)
			else:
				encode_chunk_size = getattr(self, 'retrieve_size', 64)  // 16
			self.qa_model.encode_video(video_tensor, video_id=video_id, encode_chunk_size=encode_chunk_size)

		# Process questions
		for sample in video_sample['conversations']:
			self._process_qa_sample(video_sample, sample, shift_label=shift_label, target_start_s=target_start_s)

	def _process_qa_sample(self, video_sample, sample, shift_label=None, target_start_s=None):
		"""
		Process a single QA sample. Retrieval remains default (no forced indices).
		"""
		question = sample['question']
		answer = sample.get('answer', None)

		if 'choices' in sample:  # CloseQA
			choices = sample['choices']
			if answer is None:
				answer = choices[0]
			all_yes_no = all(c.lower() in ['yes', 'no'] for c in choices)
			all_numbers = all(c.strip().isdigit() for c in choices)
			if all_yes_no or all_numbers:
				correct_choice = str(answer).strip().lower()
			else:
				try:
					correct_choice = self.choice_letters[choices.index(answer)]
				except (ValueError, IndexError):
					correct_choice = None
			prompt_kwargs = {}
			qa_results = self.video_close_qa(question, choices, correct_choice, **prompt_kwargs)
			# Add shift info into record
			qa_results = dict(qa_results)
			if shift_label is None and 'moved_to' in sample:
				shift_label = str(sample['moved_to'].get('label', 'unknown'))
			qa_results['shift_label'] = shift_label
			qa_results['target_start_s'] = target_start_s
			self._record_qa_result(video_sample, sample, qa_results, is_close_qa=True)
		else:  # OpenQA (not typical for this dataset, but keep it safe)
			qa_results = self.video_open_qa(question)
			qa_results = dict(qa_results)
			if shift_label is None and 'moved_to' in sample:
				shift_label = str(sample['moved_to'].get('label', 'unknown'))
			qa_results['shift_label'] = shift_label
			qa_results['target_start_s'] = target_start_s
			self._record_qa_result(video_sample, sample, qa_results, is_close_qa=False)


if __name__ == "__main__":
	# Reuse the generic runner
	work(ReKVOfflineVQA_TemporalShift)


