import os
import json
import argparse
import time
import torch
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="EventHallusion evaluation for ReKV")
    parser.add_argument("--save_dir", required=True, help="Directory containing prediction results")
    parser.add_argument("--annotation_path", type=str, default="all_questions.json", 
                       help="Annotation file name (entire_questions.json or misleading_questions.json)")
    parser.add_argument("--cot", action="store_true", help="Use Chain of Thought prompting")
    args = parser.parse_args()
    return args


def load_annotation(annotation_path):
    """Load EventHallusion annotation file"""
    with open(annotation_path, 'r') as file:
        ann = json.load(file)
    return ann


def evaluate_eventhal_predictions(results_path, annotation_path, cot=False):
    """
    Evaluate EventHallusion predictions against ground truth
    
    Args:
        results_path: Path to the CSV results file
        annotation_path: Path to the annotation JSON file
        cot: Whether to use Chain of Thought evaluation
    
    Returns:
        dict: Evaluation metrics
    """
    # Load predictions
    if os.path.exists(results_path):
        pred_df = pd.read_csv(results_path)
        predictions = {}
        for _, row in pred_df.iterrows():
            video_id = row['video_id']
            question = row['question']
            pred_answer = row['pred_answer']
            predictions[f"{video_id}_{question}"] = pred_answer
    else:
        print(f"Warning: Results file {results_path} not found")
        predictions = {}
    
    # Load ground truth
    ann = load_annotation(annotation_path)
    
    # Evaluation metrics
    count = 0
    total = 0
    results = []
    
    print(f"Evaluating {len(ann)} samples...")
    
    for sample_idx, sample in enumerate(tqdm(ann)):
        video_name = sample['video_id']
        
        for question_dict in sample['conversations']:
            question = question_dict['question']
            gt_answer = question_dict['answer']
            
            # Look for prediction
            pred_key = f"{video_name}_{question}"
            if pred_key in predictions:
                pred_answer = predictions[pred_key]
            else:
                # Try alternative key formats
                alt_keys = [
                    f"{video_name}_{question.lower()}",
                    f"{video_name}_{question.strip()}",
                    f"entire_{sample_idx+1:03d}_{question}"
                ]
                pred_answer = None
                for key in alt_keys:
                    if key in predictions:
                        pred_answer = predictions[key]
                        break
                
                if pred_answer is None:
                    print(f"Warning: No prediction found for {pred_key}")
                    pred_answer = ""
            
            # Evaluate correctness
            gt_clean = gt_answer.lower()
            pred_clean = pred_answer.lower()
            
            correct = False
            if gt_clean == 'yes' and 'yes' in pred_clean:
                correct = True
            elif gt_clean == 'no' and 'yes' not in pred_clean and 'no' in pred_clean:
                correct = True
            
            if correct:
                count += 1
            total += 1
            
            # Store result
            results.append({
                'video_id': video_name,
                'question': question,
                'gt_answer': gt_answer,
                'pred_answer': pred_answer,
                'correct': correct
            })
    
    # Calculate metrics
    accuracy = count / total if total > 0 else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'correct_count': count,
        'total_count': total,
        'results': results
    }
    
    return metrics


def save_evaluation_results(metrics, save_dir):
    """Save evaluation results to files"""
    # Save metrics summary
    metrics_summary = {
        'accuracy': metrics['accuracy'],
        'correct_count': metrics['correct_count'],
        'total_count': metrics['total_count']
    }
    
    with open(os.path.join(save_dir, 'eventhal_metrics.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    # Save detailed results
    with open(os.path.join(save_dir, 'eventhal_results.json'), 'w') as f:
        json.dump(metrics['results'], f, indent=2)
    
    # Save results as CSV
    results_df = pd.DataFrame(metrics['results'])
    results_df.to_csv(os.path.join(save_dir, 'eventhal_results.csv'), index=False)
    
    print(f"Results saved to {save_dir}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct_count']}/{metrics['total_count']})")


def main():
    args = parse_args()
    
    # Find annotation file
    annotation_path = os.path.join("data/eventhal", args.annotation_path)
    if not os.path.exists(annotation_path):
        print(f"Warning: Annotation file {annotation_path} not found")
        print("Please ensure the EventHallusion dataset is properly set up")
        return
    
    # Find results file
    results_path = os.path.join(args.save_dir, "results.csv")
    
    print(f"Evaluating EventHallusion predictions...")
    print(f"Results file: {results_path}")
    print(f"Annotation file: {annotation_path}")
    
    # Run evaluation
    metrics = evaluate_eventhal_predictions(results_path, annotation_path, args.cot)
    
    # Save results
    save_evaluation_results(metrics, args.save_dir)


if __name__ == "__main__":
    main() 