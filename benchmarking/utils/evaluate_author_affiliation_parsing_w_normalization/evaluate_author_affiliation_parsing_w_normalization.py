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


def calculate_metrics(true_positives: int, false_positives: int, false_negatives: int) -> EvaluationMetrics:
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
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


def evaluate_document(ground_truth: Dict, prediction: Dict) -> Dict[str, Any]:
    arxiv_id = ground_truth['arxiv_id']
    gt_authors = ground_truth['authors']

    if prediction is None or prediction.get('error'):
        return {
            'arxiv_id': arxiv_id,
            'status': 'error',
            'error': prediction.get('error') if prediction else 'Missing prediction',
            'author_metrics': calculate_metrics(0, 0, len(gt_authors)).to_dict(),
            'affiliation_metrics': calculate_metrics(0, 0, sum(len(a['affiliations']) for a in gt_authors)).to_dict()
        }
    
    pred_authors = prediction.get('predicted_authors', [])

    if pred_authors is None:
        return {
            'arxiv_id': arxiv_id,
            'status': 'null_prediction',
            'error': 'predicted_authors is None',
            'author_metrics': calculate_metrics(0, 0, len(gt_authors)).to_dict(),
            'affiliation_metrics': calculate_metrics(0, 0, sum(len(a['affiliations']) for a in gt_authors)).to_dict()
        }

    pred_authors = [a for a in pred_authors if isinstance(a, dict) and 'name' in a]

    author_eval = evaluate_author_matching(gt_authors, pred_authors)

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
        'affiliation_details': affiliation_details
    }


def evaluate_dataset(ground_truth_file: str, predictions_file: str) -> Dict[str, Any]:
    with open(ground_truth_file, 'r') as f:
        ground_truth_data = json.load(f)

    predictions = {}
    with open(predictions_file, 'r') as f:
        for line in f:
            pred = json.loads(line)
            predictions[pred['arxiv_id']] = pred

    document_results = []

    total_author_tp, total_author_fp, total_author_fn = 0, 0, 0
    total_aff_tp, total_aff_fp, total_aff_fn = 0, 0, 0
    
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

    overall_author_metrics = calculate_metrics(total_author_tp, total_author_fp, total_author_fn)
    overall_aff_metrics = calculate_metrics(total_aff_tp, total_aff_fp, total_aff_fn)

    successful_docs = [d for d in document_results if d['status'] == 'success']
    error_docs = [d for d in document_results if d['status'] == 'error']
    null_docs = [d for d in document_results if d['status'] == 'null_prediction']

    exact_matches = 0
    perfect_authors = 0
    perfect_affiliations = 0
    
    for doc in successful_docs:
        author_f1 = doc['author_evaluation']['metrics']['f1_score']
        aff_f1 = doc['affiliation_metrics']['f1_score']
        
        if author_f1 == 1.0:
            perfect_authors += 1
        if aff_f1 == 1.0:
            perfect_affiliations += 1
        if author_f1 == 1.0 and aff_f1 == 1.0:
            exact_matches += 1
    
    return {
        'summary': {
            'total_documents': len(ground_truth_data),
            'successful_predictions': len(successful_docs),
            'error_predictions': len(error_docs),
            'null_predictions': len(null_docs),
            'perfect_author_extraction': perfect_authors,
            'perfect_affiliation_extraction': perfect_affiliations,
            'perfect_both': exact_matches,
            'normalization': 'aggressive (lowercase, remove all punctuation, normalize unicode)'
        },
        'overall_metrics': {
            'author_identification': overall_author_metrics.to_dict(),
            'affiliation_extraction': overall_aff_metrics.to_dict()
        },
        'document_results': document_results
    }


def print_evaluation_report(evaluation: Dict[str, Any]):
    print("=" * 80)
    print("AFFILIATION PARSING EVALUATION - AGGRESSIVE NORMALIZATION")
    print("=" * 80)
    print()

    summary = evaluation['summary']
    print("NORMALIZATION METHOD")
    print("-" * 40)
    print(f"{summary['normalization']}")
    print()
    
    print("SUMMARY STATISTICS")
    print("-" * 40)
    print(f"Total documents: {summary['total_documents']}")
    print(f"Successful predictions: {summary['successful_predictions']}")
    print(f"Error predictions: {summary['error_predictions']}")
    print(f"Null predictions: {summary['null_predictions']}")
    print()
    
    print("PERFECT MATCHES")
    print("-" * 40)
    total = summary['total_documents']
    print(f"Perfect author extraction: {summary['perfect_author_extraction']} ({summary['perfect_author_extraction']/total*100:.1f}%)")
    print(f"Perfect affiliation extraction: {summary['perfect_affiliation_extraction']} ({summary['perfect_affiliation_extraction']/total*100:.1f}%)")
    print(f"Perfect both: {summary['perfect_both']} ({summary['perfect_both']/total*100:.1f}%)")
    print()

    print("OVERALL PERFORMANCE METRICS")
    print("-" * 40)
    
    author_metrics = evaluation['overall_metrics']['author_identification']
    print("\nAuthor Identification:")
    print(f"  Precision: {author_metrics['precision']:.4f}")
    print(f"  Recall: {author_metrics['recall']:.4f}")
    print(f"  F1-Score: {author_metrics['f1_score']:.4f}")
    print(f"  True Positives: {author_metrics['true_positives']}")
    print(f"  False Positives: {author_metrics['false_positives']}")
    print(f"  False Negatives: {author_metrics['false_negatives']}")
    
    aff_metrics = evaluation['overall_metrics']['affiliation_extraction']
    print("\nAffiliation Extraction:")
    print(f"  Precision: {aff_metrics['precision']:.4f}")
    print(f"  Recall: {aff_metrics['recall']:.4f}")
    print(f"  F1-Score: {aff_metrics['f1_score']:.4f}")
    print(f"  True Positives: {aff_metrics['true_positives']}")
    print(f"  False Positives: {aff_metrics['false_positives']}")
    print(f"  False Negatives: {aff_metrics['false_negatives']}")

    successful_docs = [d for d in evaluation['document_results'] if d['status'] == 'success']
    if successful_docs:
        sorted_docs = sorted(successful_docs, 
                           key=lambda x: (x['author_evaluation']['metrics']['f1_score'] + 
                                        x['affiliation_metrics']['f1_score']) / 2)
        
        print("\n" + "=" * 40)
        print("WORST PERFORMING DOCUMENTS (Bottom 5)")
        print("-" * 40)
        for doc in sorted_docs[:5]:
            author_f1 = doc['author_evaluation']['metrics']['f1_score']
            aff_f1 = doc['affiliation_metrics']['f1_score']
            print(f"{doc['arxiv_id']}: Author F1={author_f1:.3f}, Affiliation F1={aff_f1:.3f}")
        
        print("\n" + "=" * 40)
        print("BEST PERFORMING DOCUMENTS (Top 5)")
        print("-" * 40)
        for doc in sorted_docs[-5:]:
            author_f1 = doc['author_evaluation']['metrics']['f1_score']
            aff_f1 = doc['affiliation_metrics']['f1_score']
            print(f"{doc['arxiv_id']}: Author F1={author_f1:.3f}, Affiliation F1={aff_f1:.3f}")
    
    print("\n" + "=" * 80)


def analyze_normalization_impact(evaluation_file: str):
    """
    Analyze specific examples to show normalization impact.
    """
    with open(evaluation_file, 'r') as f:
        evaluation = json.load(f)
    
    print("\nNORMALIZATION IMPACT EXAMPLES")
    print("=" * 80)

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

    print("\nExamples of successfully matched affiliations after normalization:")
    print("-" * 40)
    for imp in improvements[:3]:
        if imp['matched']:
            print(f"\nDocument: {imp['arxiv_id']}")
            print(f"Author: {imp['author']}")
            print(f"Matched affiliations (normalized): {imp['matched'][0][:100]}...")
            if imp['missed']:
                print(f"Still missed: {len(imp['missed'])} affiliations")
            if imp['extra']:
                print(f"Extra predicted: {len(imp['extra'])} affiliations")


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
    print()

    evaluation = evaluate_dataset(args.ground_truth, args.predictions)

    with open(args.output, 'w') as f:
        json.dump(evaluation, f, indent=2)
    print(f"Evaluation results saved to: {args.output}")

    print_evaluation_report(evaluation)

    if args.analyze:
        analyze_normalization_impact(args.output)


if __name__ == '__main__':
    main()