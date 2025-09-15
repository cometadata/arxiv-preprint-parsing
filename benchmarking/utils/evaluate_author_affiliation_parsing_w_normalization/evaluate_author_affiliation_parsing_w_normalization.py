import re
import json
import argparse
import unicodedata
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict
from dataclasses import dataclass, asdict
import numpy as np
from difflib import SequenceMatcher


@dataclass
class EvaluationMetrics:
    precision: float
    recall: float
    f1_score: float
    f05_score: float 
    f15_score: float
    true_positives: int
    false_positives: int
    false_negatives: int

    def to_dict(self):
        return asdict(self)


def aggressive_normalize(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def calculate_similarity_score(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1, text2).ratio()


def normalize_author_name(name: str) -> str:
    return aggressive_normalize(name)


def calculate_f_beta(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def calculate_metrics(true_positives: int, false_positives: int, false_negatives: int) -> EvaluationMetrics:
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    f05_score = calculate_f_beta(precision, recall, 0.5)
    f15_score = calculate_f_beta(precision, recall, 1.5)

    return EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        f05_score=f05_score,
        f15_score=f15_score,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives
    )


def evaluate_author_matching(gt_authors: List[Dict], pred_authors: List[Dict]) -> Dict[str, Any]:
    gt_names = {normalize_author_name(author['name']) for author in gt_authors}

    pred_names = set()
    if pred_authors:
        for author in pred_authors:
            if author is not None and isinstance(author, dict) and 'name' in author:
                pred_names.add(normalize_author_name(author['name']))
    
    true_positives = len(gt_names & pred_names)
    false_positives = len(pred_names - gt_names)
    false_negatives = len(gt_names - pred_names)
    
    metrics = calculate_metrics(true_positives, false_positives, false_negatives)
    
    return {
        'metrics': metrics.to_dict(),
        'matched_authors': list(gt_names & pred_names),
        'missed_authors': list(gt_names - pred_names),
        'extra_authors': list(pred_names - gt_names)
    }


def evaluate_affiliations_for_author(gt_affiliations: List[str], pred_affiliations: List[str]) -> Dict[str, Any]:
    gt_affs = {aggressive_normalize(aff) for aff in gt_affiliations}
    pred_affs = {aggressive_normalize(aff) for aff in pred_affiliations}

    true_positives = len(gt_affs & pred_affs)
    false_positives = len(pred_affs - gt_affs)
    false_negatives = len(gt_affs - pred_affs)

    metrics = calculate_metrics(true_positives, false_positives, false_negatives)

    return {
        'metrics': metrics.to_dict(),
        'exact_match': gt_affs == pred_affs,
        'matched': list(gt_affs & pred_affs),
        'missed': list(gt_affs - pred_affs),
        'extra': list(pred_affs - gt_affs)
    }


def evaluate_combined_author_affiliations(gt_authors: List[Dict], pred_authors: List[Dict]) -> Dict[str, Any]:
    gt_author_dict = {}
    for author in gt_authors:
        norm_name = normalize_author_name(author['name'])
        gt_author_dict[norm_name] = {aggressive_normalize(aff) for aff in author['affiliations']}

    pred_author_dict = {}
    if pred_authors:
        for author in pred_authors:
            if author is not None and isinstance(author, dict) and 'name' in author:
                norm_name = normalize_author_name(author['name'])
                pred_author_dict[norm_name] = {aggressive_normalize(aff) for aff in author.get('affiliations', [])}

    true_positives = 0
    partially_correct = []
    fully_correct = []

    for author_name in gt_author_dict:
        if author_name in pred_author_dict:
            gt_affs = gt_author_dict[author_name]
            pred_affs = pred_author_dict[author_name]

            if gt_affs == pred_affs:
                true_positives += 1
                fully_correct.append(author_name)
            else:
                partially_correct.append({
                    'author': author_name,
                    'matched_affs': len(gt_affs & pred_affs),
                    'total_affs': len(gt_affs),
                    'extra_affs': len(pred_affs - gt_affs)
                })

    false_positives = len(pred_author_dict) - true_positives

    false_negatives = len(gt_author_dict) - true_positives

    metrics = calculate_metrics(true_positives, false_positives, false_negatives)

    return {
        'metrics': metrics.to_dict(),
        'fully_correct_authors': fully_correct,
        'partially_correct_authors': partially_correct,
        'total_gt_authors': len(gt_author_dict),
        'total_pred_authors': len(pred_author_dict)
    }


def evaluate_document(ground_truth: Dict, prediction: Dict) -> Dict[str, Any]:
    arxiv_id = ground_truth['arxiv_id']
    gt_authors = ground_truth['authors']

    if prediction is None or prediction.get('error'):
        return {
            'arxiv_id': arxiv_id,
            'status': 'error',
            'error': prediction.get('error') if prediction else 'Missing prediction',
            'author_metrics': calculate_metrics(0, 0, len(gt_authors)).to_dict(),
            'affiliation_metrics': calculate_metrics(0, 0, sum(len(a['affiliations']) for a in gt_authors)).to_dict(),
            'combined_metrics': calculate_metrics(0, 0, len(gt_authors)).to_dict()
        }

    pred_authors = prediction.get('predicted_authors', [])

    if pred_authors is None:
        return {
            'arxiv_id': arxiv_id,
            'status': 'null_prediction',
            'error': 'predicted_authors is None',
            'author_metrics': calculate_metrics(0, 0, len(gt_authors)).to_dict(),
            'affiliation_metrics': calculate_metrics(0, 0, sum(len(a['affiliations']) for a in gt_authors)).to_dict(),
            'combined_metrics': calculate_metrics(0, 0, len(gt_authors)).to_dict()
        }

    pred_authors = [a for a in pred_authors if isinstance(a, dict) and 'name' in a]

    author_eval = evaluate_author_matching(gt_authors, pred_authors)
    combined_eval = evaluate_combined_author_affiliations(gt_authors, pred_authors)

    gt_author_dict = {normalize_author_name(a['name']): a['affiliations'] for a in gt_authors}
    pred_author_dict = {}
    for a in pred_authors:
        if a is not None and isinstance(a, dict) and 'name' in a:
            pred_author_dict[normalize_author_name(a['name'])] = a.get('affiliations', [])

    affiliation_details = []
    total_aff_tp, total_aff_fp, total_aff_fn = 0, 0, 0
    
    for author_name in author_eval['matched_authors']:
        gt_affs = gt_author_dict.get(author_name, [])
        pred_affs = pred_author_dict.get(author_name, [])
        
        aff_eval = evaluate_affiliations_for_author(gt_affs, pred_affs)
        affiliation_details.append({
            'author': author_name,
            'evaluation': aff_eval
        })
        
        total_aff_tp += aff_eval['metrics']['true_positives']
        total_aff_fp += aff_eval['metrics']['false_positives']
        total_aff_fn += aff_eval['metrics']['false_negatives']

    for missed_author in author_eval['missed_authors']:
        gt_affs = gt_author_dict.get(missed_author, [])
        total_aff_fn += len(gt_affs)

    for extra_author in author_eval['extra_authors']:
        pred_affs = pred_author_dict.get(extra_author, [])
        total_aff_fp += len(pred_affs)
    
    affiliation_metrics = calculate_metrics(total_aff_tp, total_aff_fp, total_aff_fn)

    return {
        'arxiv_id': arxiv_id,
        'status': 'success',
        'author_evaluation': author_eval,
        'affiliation_metrics': affiliation_metrics.to_dict(),
        'affiliation_details': affiliation_details,
        'combined_evaluation': combined_eval
    }


def evaluate_dataset(ground_truth_file: str, predictions_file: str) -> Dict[str, Any]:
    with open(ground_truth_file, 'r') as f:
        ground_truth_data = json.load(f)

    with open(predictions_file, 'r') as f:
        predictions_data = json.load(f)

    predictions = {}
    for gt_doc in ground_truth_data:
        gt_arxiv_id = gt_doc['arxiv_id']

        if gt_arxiv_id in predictions_data:
            predictions[gt_arxiv_id] = {
                'arxiv_id': gt_arxiv_id,
                'predicted_authors': predictions_data[gt_arxiv_id]
            }
        else:
            normalized_id = gt_arxiv_id.replace('arXiv:', '') if gt_arxiv_id.startswith('arXiv:') else gt_arxiv_id
            normalized_id = normalized_id.replace('/', '_')
            if normalized_id in predictions_data:
                predictions[gt_arxiv_id] = {
                    'arxiv_id': gt_arxiv_id,
                    'predicted_authors': predictions_data[normalized_id]
                }

    document_results = []

    total_author_tp, total_author_fp, total_author_fn = 0, 0, 0
    total_aff_tp, total_aff_fp, total_aff_fn = 0, 0, 0
    total_combined_tp, total_combined_fp, total_combined_fn = 0, 0, 0

    for gt_doc in ground_truth_data:
        arxiv_id = gt_doc['arxiv_id']
        pred_doc = predictions.get(arxiv_id)

        doc_eval = evaluate_document(gt_doc, pred_doc)
        document_results.append(doc_eval)

        if doc_eval['status'] == 'success':
            author_metrics = doc_eval['author_evaluation']['metrics']
            total_author_tp += author_metrics['true_positives']
            total_author_fp += author_metrics['false_positives']
            total_author_fn += author_metrics['false_negatives']

            aff_metrics = doc_eval['affiliation_metrics']
            total_aff_tp += aff_metrics['true_positives']
            total_aff_fp += aff_metrics['false_positives']
            total_aff_fn += aff_metrics['false_negatives']

            combined_metrics = doc_eval['combined_evaluation']['metrics']
            total_combined_tp += combined_metrics['true_positives']
            total_combined_fp += combined_metrics['false_positives']
            total_combined_fn += combined_metrics['false_negatives']

    overall_author_metrics = calculate_metrics(total_author_tp, total_author_fp, total_author_fn)
    overall_aff_metrics = calculate_metrics(total_aff_tp, total_aff_fp, total_aff_fn)
    overall_combined_metrics = calculate_metrics(total_combined_tp, total_combined_fp, total_combined_fn)

    successful_docs = [d for d in document_results if d['status'] == 'success']
    error_docs = [d for d in document_results if d['status'] == 'error']
    null_docs = [d for d in document_results if d['status'] == 'null_prediction']

    exact_matches = 0
    perfect_authors = 0
    perfect_affiliations = 0
    perfect_combined = 0

    for doc in successful_docs:
        author_f1 = doc['author_evaluation']['metrics']['f1_score']
        aff_f1 = doc['affiliation_metrics']['f1_score']
        combined_f1 = doc['combined_evaluation']['metrics']['f1_score']

        if author_f1 == 1.0:
            perfect_authors += 1
        if aff_f1 == 1.0:
            perfect_affiliations += 1
        if author_f1 == 1.0 and aff_f1 == 1.0:
            exact_matches += 1
        if combined_f1 == 1.0:
            perfect_combined += 1

    return {
        'summary': {
            'total_documents': len(ground_truth_data),
            'successful_predictions': len(successful_docs),
            'error_predictions': len(error_docs),
            'null_predictions': len(null_docs),
            'perfect_author_extraction': perfect_authors,
            'perfect_affiliation_extraction': perfect_affiliations,
            'perfect_both': exact_matches,
            'perfect_combined_extraction': perfect_combined,
            'normalization': 'aggressive (lowercase, remove all punctuation, normalize unicode)'
        },
        'overall_metrics': {
            'author_identification': overall_author_metrics.to_dict(),
            'affiliation_extraction': overall_aff_metrics.to_dict(),
            'combined_author_affiliation': overall_combined_metrics.to_dict()
        },
        'document_results': document_results
    }


def analyze_normalization_impact(evaluation_file: str):
    with open(evaluation_file, 'r') as f:
        evaluation = json.load(f)

    improvements = []
    for doc in evaluation['document_results']:
        if doc['status'] == 'success' and doc['affiliation_details']:
            for detail in doc['affiliation_details']:
                if detail['evaluation']['metrics']['true_positives'] > 0:
                    improvements.append({
                        'arxiv_id': doc['arxiv_id'],
                        'author': detail['author'],
                        'matched': detail['evaluation']['matched'],
                        'missed': detail['evaluation']['missed'],
                        'extra': detail['evaluation']['extra']
                    })


def main():
    parser = argparse.ArgumentParser(description='Evaluate affiliation parsing with aggressive normalization')
    parser.add_argument( '-g', '--ground-truth',
                       required=True,
                       help='Path to ground truth JSON file')
    parser.add_argument( '-p','--predictions',
                       required=True,
                       help='Path to predictions JSONL file')
    parser.add_argument('--output', '-o',
                       default='evaluation_aggressive_norm.json',
                       help='Path to save evaluation results')
    parser.add_argument('--analyze', '-a', action='store_true',
                       help='Show normalization impact analysis')
    
    args = parser.parse_args()
    
    print(f"Evaluating predictions with aggressive normalization")
    print(f"Predictions: {args.predictions}")
    print(f"Ground truth: {args.ground_truth}")

    evaluation = evaluate_dataset(args.ground_truth, args.predictions)

    with open(args.output, 'w') as f:
        json.dump(evaluation, f, indent=2)
    print(f"Evaluation results saved to: {args.output}")

    if args.analyze:
        analyze_normalization_impact(args.output)


if __name__ == '__main__':
    main()