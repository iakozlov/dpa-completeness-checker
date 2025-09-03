#!/usr/bin/env python3
"""
aggregate_evaluations.py - Aggregate evaluation results from multiple DPAs
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Any

def calculate_metrics_from_confusion_matrix(tp: int, fp: int, tn: int, fn: int) -> Dict:
    """Calculate accuracy, precision, recall, and F1 score from confusion matrix values.
    
    Args:
        tp: True positives
        fp: False positives
        tn: True negatives
        fn: False negatives
        
    Returns:
        Dict containing accuracy, precision, recall, and F1 score
    """
    # Avoid division by zero
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3)
    }

def calculate_aggregated_metrics(confusion_matrices: List[Dict]) -> Dict:
    """Calculate aggregated metrics from aggregated confusion matrix values.
    
    Args:
        confusion_matrices: List of confusion matrices to aggregate
        
    Returns:
        Dictionary with correctly calculated aggregated metrics
    """
    if not confusion_matrices:
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0
        }
    
    # Aggregate confusion matrix values
    total_tp = sum(matrix.get("true_positives", 0) for matrix in confusion_matrices)
    total_fp = sum(matrix.get("false_positives", 0) for matrix in confusion_matrices)
    total_tn = sum(matrix.get("true_negatives", 0) for matrix in confusion_matrices)
    total_fn = sum(matrix.get("false_negatives", 0) for matrix in confusion_matrices)
    
    # Calculate metrics from aggregated confusion matrix
    return calculate_metrics_from_confusion_matrix(total_tp, total_fp, total_tn, total_fn)

def aggregate_confusion_matrices(matrices: List[Dict]) -> Dict:
    """Aggregate confusion matrices by summing their values.
    
    Args:
        matrices: List of confusion matrices
        
    Returns:
        Aggregated confusion matrix
    """
    if not matrices:
        return {
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0
        }
    
    result = defaultdict(int)
    for matrix in matrices:
        for key, value in matrix.items():
            result[key] += value
    
    return dict(result)

def get_consistent_req_mapping():
    """Return a consistent mapping of requirement IDs to requirement numbers."""
    return {
        # R10-R29 to correct requirement numbers
        "10": "1",
        "11": "2",
        "12": "3",
        "13": "4",
        # R14 doesn't exist
        "15": "5",
        "16": "6",
        "17": "7",
        "18": "8",  # Important: 18 maps to 8, not 9
        "19": "9",
        "20": "10",
        "21": "12",  # Fixed: R21 maps to requirement 12 (consulting supervisory authorities)
        "22": "11",  # Fixed: R22 maps to requirement 11 (data protection impact assessment)
        "23": "13",
        "24": "14",
        "25": "15",
        "26": "16",
        "27": "17",
        "28": "18",  # This should be mapped consistently with evaluate_completeness.py
        "29": "19",
    }

def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results from multiple DPAs")
    parser.add_argument("--input_files", type=str, required=True, help="Space-separated list of input JSON files")
    parser.add_argument("--output", type=str, required=True, help="Output file for aggregated results")
    
    args = parser.parse_args()
    
    # Parse input files
    input_files = []
    if args.input_files:
        # Handle both space-separated and comma-separated lists
        if ',' in args.input_files:
            input_files = [f.strip() for f in args.input_files.split(',')]
        else:
            input_files = [f.strip() for f in args.input_files.split()]
    
    if not input_files:
        print("Error: No input files specified")
        return
    
    print(f"Aggregating evaluation results from {len(input_files)} files")
    
    # Get the consistent requirement mapping
    req_mapping = get_consistent_req_mapping()
    
    # Load all evaluation results
    all_evaluations = []
    for file_path in input_files:
        try:
            with open(file_path, 'r') as f:
                eval_result = json.load(f)
                
                # Fix any inconsistent requirement_number in requirements_details
                if "requirements_details" in eval_result:
                    for req_id, details in eval_result["requirements_details"].items():
                        if req_id in req_mapping:
                            details["requirement_number"] = req_mapping[req_id]
                
                all_evaluations.append(eval_result)
            print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_evaluations:
        print("No valid evaluation files found. Exiting.")
        return
    
    # Extract DPA names for reference
    dpa_names = [eval_result.get("dpa", f"Unknown DPA {i}") for i, eval_result in enumerate(all_evaluations)]
    
    # Initialize aggregated result structure
    aggregated_result = {
        "dpas": dpa_names,
        "total_evaluations": len(all_evaluations),
        "segments_evaluated": sum(eval_result.get("segments_evaluated", 0) for eval_result in all_evaluations),
        "requirements_evaluated": all_evaluations[0].get("requirements_evaluated", []),
        "aggregated_metrics": {},
        "per_dpa_results": {}
    }
    
    # Store individual DPA results for reference
    for i, dpa in enumerate(dpa_names):
        aggregated_result["per_dpa_results"][dpa] = all_evaluations[i]
    
    # Aggregate requirements metrics
    req_metrics_list = [eval_result.get("requirements_metrics", {}).get("overall", {}) 
                       for eval_result in all_evaluations]
    req_confusion_matrices = [eval_result.get("requirements_metrics", {}).get("confusion_matrix", {}) 
                             for eval_result in all_evaluations]
    
    aggregated_result["aggregated_metrics"]["requirements"] = {
        "overall": calculate_aggregated_metrics(req_confusion_matrices),
        "confusion_matrix": aggregate_confusion_matrices(req_confusion_matrices)
    }
    
    # Aggregate segment metrics
    seg_metrics_list = [eval_result.get("segment_metrics", {}).get("overall", {}) 
                       for eval_result in all_evaluations]
    seg_confusion_matrices = [eval_result.get("segment_metrics", {}).get("confusion_matrix", {}) 
                             for eval_result in all_evaluations]
    
    aggregated_result["aggregated_metrics"]["segments"] = {
        "overall": calculate_aggregated_metrics(seg_confusion_matrices),
        "confusion_matrix": aggregate_confusion_matrices(seg_confusion_matrices)
    }
    
    # Compute aggregated completeness
    complete_ground_truth = sum(1 for eval_result in all_evaluations 
                               if eval_result.get("ground_truth", {}).get("is_complete", False))
    complete_prediction = sum(1 for eval_result in all_evaluations 
                             if eval_result.get("prediction", {}).get("is_complete", False))
    
    aggregated_result["completeness_summary"] = {
        "ground_truth_complete": complete_ground_truth,
        "prediction_complete": complete_prediction,
        "ground_truth_percentage": round(complete_ground_truth / len(all_evaluations) * 100, 1),
        "prediction_percentage": round(complete_prediction / len(all_evaluations) * 100, 1),
    }
    
    # Save aggregated results
    with open(args.output, 'w') as f:
        json.dump(aggregated_result, f, indent=2)
    
    # Print summary
    print("\nAggregated Evaluation Summary:")
    print(f"Total DPAs evaluated: {len(dpa_names)}")
    print(f"DPAs evaluated: {', '.join(dpa_names)}")
    print(f"Total segments evaluated: {aggregated_result['segments_evaluated']}")
    
    req_cm = aggregated_result["aggregated_metrics"]["requirements"]["confusion_matrix"]
    print("\nAggregated Requirement-Level Metrics:")
    print(f"Confusion Matrix: TP={req_cm['true_positives']}, FP={req_cm['false_positives']}, " 
          f"TN={req_cm['true_negatives']}, FN={req_cm['false_negatives']}")
    
    for metric, value in aggregated_result["aggregated_metrics"]["requirements"]["overall"].items():
        print(f"  {metric.capitalize()}: {value}")
    
    seg_cm = aggregated_result["aggregated_metrics"]["segments"]["confusion_matrix"]
    print("\nAggregated Segment-Level Metrics:")
    print(f"Confusion Matrix: TP={seg_cm['true_positives']}, FP={seg_cm['false_positives']}, " 
          f"TN={seg_cm['true_negatives']}, FN={seg_cm['false_negatives']}")
    
    for metric, value in aggregated_result["aggregated_metrics"]["segments"]["overall"].items():
        print(f"  {metric.capitalize()}: {value}")
    
    print(f"\nCompleteness Summary:")
    comp_summary = aggregated_result["completeness_summary"]
    print(f"  Ground Truth Complete: {comp_summary['ground_truth_complete']} of {len(all_evaluations)} " 
          f"({comp_summary['ground_truth_percentage']}%)")
    print(f"  Prediction Complete: {comp_summary['prediction_complete']} of {len(all_evaluations)} " 
          f"({comp_summary['prediction_percentage']}%)")
    
    print(f"\nAggregated results saved to: {args.output}")

if __name__ == "__main__":
    main() 