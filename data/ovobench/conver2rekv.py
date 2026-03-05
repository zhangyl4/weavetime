import json
from collections import defaultdict, Counter
from logzero import logger
import os

def convert_ovo_to_rekv(input_path, output_path):
    """Convert OVO benchmark format to ReKV format.
    
    Args:
        input_path: Path to OVO benchmark JSON file
        output_path: Path to save converted ReKV format JSON file
    """
    # Read input JSON
    with open(input_path, 'r') as f:
        ovo_data = json.load(f)

    # Group by video
    video_groups = defaultdict(list)
    task_counter = Counter()
    for item in ovo_data:
        # Handle SSR, CRR, REC tasks
        if item.get('task') in ['SSR', 'CRR', 'REC']:
            if item['task'] == 'SSR':
                # Format SSR task
                for test_info in item['test_info']:
                    formatted_item = {
                        'id': item['id'],
                        'task': item['task'],
                        'video': item['video'],
                        'realtime': test_info['realtime'],
                        'question': f"""You're watching a tutorial video which contain a sequential of steps.
The following is one step from the whole procedures:
{test_info['step']}
Your task is to decide: Is the man/woman in the video currently carrying out this step?
Return "Yes" if the man/woman in the video is currently performing this step;
Return "No" if not.""",
                        'answer': 'Yes' if test_info['type'] == 1 else 'No',
                        'gt': test_info['type']
                    }
                    video_groups[item['video']].append(formatted_item)
                    task_counter[item['task']] += 1
            elif item['task'] == 'CRR':
                # Format CRR task
                for test_info in item['test_info']:
                    formatted_item = {
                        'id': item['id'],
                        'task': item['task'],
                        'video': item['video'],
                        'realtime': test_info['realtime'],
                        'question': f"""You're responsible of answering questions based on the video content.
The following question are relevant to the latest frames, i.e. the end of the video.
{item['question']}
Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response.""",
                        'answer': 'Yes' if test_info['type'] == 1 else 'No',
                        'gt': test_info['type']
                    }
                    video_groups[item['video']].append(formatted_item)
                    task_counter[item['task']] += 1
            elif item['task'] == 'REC':
                # Format REC task
                for test_info in item['test_info']:
                    formatted_item = {
                        'id': item['id'],
                        'task': item['task'],
                        'video': item['video'],
                        'realtime': test_info['realtime'],
                        'question': f"""You're watching a video in which people may perform a certain type of action repetively.
The person performing this kind of action are referred to as 'they' in the following statement.
You're task is to count how many times have different people in the video perform this kind of action in total.
One complete motion counts as one. 
Now, answer the following question: {item['activity']}
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response.""",
                        'answer': str(test_info['count']),
                        'gt': test_info['count']
                    }
                    video_groups[item['video']].append(formatted_item)
                    task_counter[item['task']] += 1
        else:
            # Handle other tasks (original format)
            formatted_item = {
                'id': item['id'],
                'task': item['task'],
                'video': item['video'],
                'realtime': item['realtime'],
                'question': item['question'],
                'answer': item['options'][item['gt']],
                'choices': item['options'],
                'gt': item['gt']
            }
            video_groups[item['video']].append(formatted_item)
            task_counter[item['task']] += 1

    # Convert to ReKV format
    rekv_data = []
    for video_path, items in video_groups.items():
        # Sort items by realtime
        try:
            items.sort(key=lambda x: x['realtime'])
        except:
            breakpoint()
        
        # Create conversations list
        conversations = []
        prev_time = 0
        for item in items:
            conv = {
                'question': item['question'],
                'answer': item['answer'],
                'start_time': prev_time,
                'end_time': item['realtime'],
                'task': item['task']
            }
            if 'choices' in item:
                conv['choices'] = item['choices']
            conversations.append(conv)
            prev_time = item['realtime']

        # Create video entry
        video_entry = {
            'video_id': video_path.split('/')[-1].split('.')[0],
            'video_path': os.path.join('data/ovobench/videos', video_path),
            'conversations': conversations
        }
        rekv_data.append(video_entry)

    # Save output JSON
    with open(output_path, 'w') as f:
        json.dump(rekv_data, f, indent=2)
    
    logger.info(f'Converted {len(ovo_data)} items into {len(rekv_data)} video entries')

    # Print per-task question counts
    logger.info('Task question counts:')
    for task, count in task_counter.items():
        logger.info(f'Task {task}: {count} questions')

if __name__ == '__main__':
    input_path = '/root/videollm-online/ReKV/data/ovobench/ovo_bench_new.json'
    output_path = '/root/videollm-online/ReKV/data/ovobench/ovo_bench_rekv.json'
    convert_ovo_to_rekv(input_path, output_path)
