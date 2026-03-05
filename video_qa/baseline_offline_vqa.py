import torch
from logzero import logger

from video_qa.base import BaseVQA, work


class ReKVOfflineVQA(BaseVQA):
    def video_open_qa(self, question, max_new_tokens=1024, retrieved_indices=None):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }
        # pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens, retrieved_indices=retrieved_indices)
        # 使用新的方法获取QA结果和retrieval信息
        pred_answer, retrieval_info = self.capture_retrieval_info_and_qa(input_text, max_new_tokens=max_new_tokens, retrieved_indices=retrieved_indices)

        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'retrieval_info': retrieval_info,
        }

    def video_close_qa(self, question, candidates, correct_choice, retrieved_indices=None):
        input_text = self.format_mcqa_prompt(question, candidates)
        # pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16, retrieved_indices=retrieved_indices)
        # 使用新的方法获取QA结果和retrieval信息
        pred_answer, retrieval_info = self.capture_retrieval_info_and_qa(input_text, max_new_tokens=16, retrieved_indices=retrieved_indices)
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
            'retrieval_info': retrieval_info,
        }

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        # load and preprocess video frames for QA
        video_path = video_sample['video_path']
        video = self.load_video(video_path, max_frame_num=self.retrieve_size) # baseline with uniform sample
        if not isinstance(video, torch.Tensor):
            video_tensor = torch.from_numpy(video)
        else:
            video_tensor = video

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()
        
        # 使用视频路径作为video_id以确保固定分辨率
        video_id = video_path
        self.qa_model.encode_video(video_tensor, video_id=video_id) # , encode_chunk_size=self.retrieve_size

        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            question = sample['question']
            answer = sample['answer']
            
            # QA
            if 'choices' in sample:  # CloseQA
                choices = sample['choices']
                if answer is None:  # FIXME: an ugly fix for some benchmarks do not provide GT
                    answer = choices[0]
                correct_choice = self.choice_letters[choices.index(answer)]
                qa_results = self.video_close_qa(question, choices, correct_choice)
                
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
                qa_results = self.video_open_qa(question)
                
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

            if 'question_type' in sample:
                self.record[(self.retrieve_size, self.chunk_size)][-1]['task'] = sample['question_type']


if __name__ == "__main__":
    work(ReKVOfflineVQA)
