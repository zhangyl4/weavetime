import torch
import numpy as np
from logzero import logger
from decord import VideoReader, cpu

from video_qa.base import BaseVQA, work

import math

def ceil_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(math.ceil(time * fps) / fps, min_time), max_time)

def floor_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(math.floor(time * fps) / fps, min_time), max_time)

class ReKVStreamVQA(BaseVQA):

    def video_open_qa(self, video, question, max_new_tokens=1024):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }
        pred_answer = self.qa_model.simple_forward(video, input_text['question'])

        return {
            'pred_answer': pred_answer.replace('\n', ''),
        }
        
    
    def video_close_qa(self, video, question, candidates, correct_choice):
        """Perform multiple-choice question answering."""
        input_text = self.format_mcqa_prompt(question, candidates)
        pred_answer = self.qa_model.simple_forward(video, input_text['formatted_question'])
        
        pred_letter = self.extract_characters_regex(pred_answer)
        
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    
    def question_answer(self, video_sample, sample, video):
        question = sample['question']
        answer = sample['answer']
        # QA
        if 'choices' in sample:  # CloseQA
            choices = sample['choices']
            if answer is None:  # FIXME: an ugly fix for some benchmarks do not provide GT
                answer = choices[0]

            correct_choice = self.choice_letters[choices.index(answer)]
            
            qa_results = self.video_close_qa(video, question, choices, correct_choice)
            # 从QA结果中提取retrieval信息
            retrieval_info = qa_results.get('retrieval_info', {})
            retrieval_records = retrieval_info.get('retrieval_records', {})
            similarity_scores = retrieval_info.get('similarity_scores', {})
            
            self.record[(self.retrieve_size, self.chunk_size)].append({
                'video_id': video_sample['video_id'],
                'question': question,
                'choices': choices,
                'answer': answer,
                'correct_choice': correct_choice,
                'pred_answer': qa_results['pred_answer'],
                'pred_choice': qa_results['pred_choice'],
                'qa_acc': qa_results['acc'] * 100,
                'retrieval_records': retrieval_records,
                'similarity_scores': similarity_scores,
            })
        else:  # OpenQA
            qa_results = self.video_open_qa(video, question)
            
            # 从QA结果中提取retrieval信息
            retrieval_info = qa_results.get('retrieval_info', {})
            retrieval_records = retrieval_info.get('retrieval_records', {})
            similarity_scores = retrieval_info.get('similarity_scores', {})
            
            self.record[(self.retrieve_size, self.chunk_size)].append({
                'video_id': video_sample['video_id'],
                'question': question,
                'answer': answer,
                'pred_answer': qa_results['pred_answer'],
                'retrieval_records': retrieval_records,
                'similarity_scores': similarity_scores,
            })

        if 'task' in sample:
            self.record[(self.retrieve_size, self.chunk_size)][-1]['task'] = sample['task']

    
    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        # load and preprocess video frames for QA
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
        
        for sample in video_sample['conversations']:
            # forward question by question

            # 使用video_sample中的video_id确保固定分辨率
            video_id = video_sample.get('video_id', video_sample.get('video_path', 'unknown'))
            
            logger.debug(f'sample: {sample}')

            # get clip start and end index
            start_time_idx = floor_time_by_fps(sample['start_time'], self.sample_fps, 0, 9999999) * self.sample_fps
            end_time_idx = min(ceil_time_by_fps(sample['end_time'], self.sample_fps, 0, 9999) * self.sample_fps, video_tensor.shape[0] - 1) * self.sample_fps
            start_time_idx = int(start_time_idx)
            end_time_idx = int(end_time_idx)
            # force sample
            frame_idx = range(0, end_time_idx+1)
            if len(frame_idx) > self.retrieve_size:
                frame_idx = np.linspace(0, end_time_idx, self.retrieve_size, dtype=int).tolist()
            
            # QA
            self.question_answer(video_sample, sample, video_tensor[frame_idx])


if __name__ == "__main__":
    work(ReKVStreamVQA)
