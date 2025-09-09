import re
import json
import argparse
import unicodedata
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple


def aggressive_normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def compare_affiliations(gt_affiliations: List[str], pred_affiliations: List[str]) -> Dict[str, List[str]]:
    gt_norm_to_orig = {aggressive_normalize(
        aff): aff for aff in gt_affiliations}
    pred_norm_to_orig = {aggressive_normalize(
        aff): aff for aff in pred_affiliations}
    gt_normalized = set(gt_norm_to_orig.keys())
    pred_normalized = set(pred_norm_to_orig.keys())
    matched_norm = gt_normalized & pred_normalized
    missed_norm = gt_normalized - pred_normalized
    extra_norm = pred_normalized - gt_normalized

    return {
        'matched': [gt_norm_to_orig[norm] for norm in matched_norm],
        'missed': [gt_norm_to_orig[norm] for norm in missed_norm],
        'extra': [pred_norm_to_orig[norm] for norm in extra_norm],
        'matched_normalized': list(matched_norm),
        'missed_normalized': list(missed_norm),
        'extra_normalized': list(extra_norm)
    }


def analyze_discrepancies_normalized(evaluation_file: str):
    with open(evaluation_file, 'r') as f:
        evaluation = json.load(f)
    all_missed_authors = []
    all_extra_authors = []
    all_affiliation_mismatches = []
    missed_affiliations_normalized = Counter()
    extra_affiliations_normalized = Counter()
    missed_affiliations_original = []
    extra_affiliations_original = []
    docs_with_perfect_authors = 0
    docs_with_perfect_affiliations = 0
    docs_with_both_perfect = 0
    docs_with_formatting_only_issues = 0

    for doc in evaluation['document_results']:
        arxiv_id = doc['arxiv_id']

        if doc['status'] == 'success':
            author_f1 = doc.get('author_evaluation', {}).get(
                'metrics', {}).get('f1_score', 0)
            aff_f1 = doc.get('affiliation_metrics', {}).get('f1_score', 0)
            if author_f1 == 1.0:
                docs_with_perfect_authors += 1
            if aff_f1 == 1.0:
                docs_with_perfect_affiliations += 1
            if author_f1 == 1.0 and aff_f1 == 1.0:
                docs_with_both_perfect += 1

            discrepancies = doc.get('discrepancies', {})

            if not discrepancies:
                author_eval = doc.get('author_evaluation', {})
                missed_authors_names = author_eval.get('missed_authors', [])
                extra_authors_names = author_eval.get('extra_authors', [])

                for author_name in missed_authors_names:
                    discrepancies.setdefault('missed_authors', []).append({
                        'name': author_name,
                        'affiliations': []
                    })

                for author_name in extra_authors_names:
                    discrepancies.setdefault('extra_authors', []).append({
                        'name': author_name,
                        'affiliations': []
                    })

                affiliation_details = doc.get('affiliation_details', [])
                for detail in affiliation_details:
                    if 'evaluation' in detail:
                        eval_data = detail['evaluation']
                        if eval_data.get('missed') or eval_data.get('extra'):
                            discrepancies.setdefault('affiliation_mismatches', []).append({
                                'author': detail['author'],
                                'ground_truth_affiliations': [],
                                'predicted_affiliations': [],
                                'missed': eval_data.get('missed', []),
                                'extra': eval_data.get('extra', []),
                                'matched': eval_data.get('matched', [])
                            })

            for author in discrepancies.get('missed_authors', []):
                normalized_name = aggressive_normalize(author['name'])
                all_missed_authors.append({
                    'arxiv_id': arxiv_id,
                    'author_name': author['name'],
                    'author_name_normalized': normalized_name,
                    'affiliations': author['affiliations']
                })

            for author in discrepancies.get('extra_authors', []):
                normalized_name = aggressive_normalize(author['name'])
                all_extra_authors.append({
                    'arxiv_id': arxiv_id,
                    'author_name': author['name'],
                    'author_name_normalized': normalized_name,
                    'affiliations': author['affiliations']
                })

            for mismatch in discrepancies.get('affiliation_mismatches', []):
                if mismatch.get('ground_truth_affiliations') and mismatch.get('predicted_affiliations'):
                    comparison = compare_affiliations(
                        mismatch['ground_truth_affiliations'],
                        mismatch['predicted_affiliations']
                    )
                    ground_truth = mismatch['ground_truth_affiliations']
                    predicted = mismatch['predicted_affiliations']
                else:
                    comparison = {
                        'matched': mismatch.get('matched', []),
                        'missed': mismatch.get('missed', []),
                        'extra': mismatch.get('extra', []),
                        'matched_normalized': [],  # Already normalized
                        'missed_normalized': mismatch.get('missed', []),
                        'extra_normalized': mismatch.get('extra', [])
                    }
                    ground_truth = []
                    predicted = []

                has_real_mismatches = len(comparison.get('missed', [])) > 0 or len(
                    comparison.get('extra', [])) > 0

                if not has_real_mismatches and ground_truth and predicted and len(ground_truth) == len(predicted):
                    docs_with_formatting_only_issues += 1

                all_affiliation_mismatches.append({
                    'arxiv_id': arxiv_id,
                    'author': mismatch['author'],
                    'ground_truth': ground_truth,
                    'predicted': predicted,
                    'matched_after_norm': comparison.get('matched', []),
                    'missed_after_norm': comparison.get('missed', []),
                    'extra_after_norm': comparison.get('extra', []),
                    'is_formatting_only': not has_real_mismatches
                })

                for missed in comparison.get('missed', []):
                    if missed:
                        missed_affiliations_original.append(missed)
                        if re.search(r'[^\w\s]', missed):
                            missed_affiliations_normalized[aggressive_normalize(
                                missed)] += 1
                        else:
                            missed_affiliations_normalized[missed] += 1

                for extra in comparison.get('extra', []):
                    if extra:
                        extra_affiliations_original.append(extra)
                        if re.search(r'[^\w\s]', extra):
                            extra_affiliations_normalized[aggressive_normalize(
                                extra)] += 1
                        else:
                            extra_affiliations_normalized[extra] += 1

    common_missed_normalized = missed_affiliations_normalized.most_common(20)
    common_extra_normalized = extra_affiliations_normalized.most_common(20)

    norm_to_original_missed = defaultdict(list)
    for orig in missed_affiliations_original:
        norm = aggressive_normalize(orig)
        if orig not in norm_to_original_missed[norm]:
            norm_to_original_missed[norm].append(orig)

    norm_to_original_extra = defaultdict(list)
    for orig in extra_affiliations_original:
        norm = aggressive_normalize(orig)
        if orig not in norm_to_original_extra[norm]:
            norm_to_original_extra[norm].append(orig)

    summary_stats = {
        'total_documents': len(evaluation['document_results']),
        'documents_with_perfect_authors': docs_with_perfect_authors,
        'documents_with_perfect_affiliations': docs_with_perfect_affiliations,
        'documents_with_both_perfect': docs_with_both_perfect,
        'documents_with_formatting_only_issues': docs_with_formatting_only_issues,
        'total_missed_authors': len(all_missed_authors),
        'total_extra_authors': len(all_extra_authors),
        'total_affiliation_mismatches': len(all_affiliation_mismatches),
        'affiliation_mismatches_formatting_only': sum(1 for m in all_affiliation_mismatches if m['is_formatting_only']),
        'affiliation_mismatches_real': sum(1 for m in all_affiliation_mismatches if not m['is_formatting_only']),
        'unique_missed_author_names': len(set(a['author_name_normalized'] for a in all_missed_authors)),
        'unique_extra_author_names': len(set(a['author_name_normalized'] for a in all_extra_authors))
    }

    formatting_only_mismatches = [
        m for m in all_affiliation_mismatches if m['is_formatting_only']]
    real_mismatches = [
        m for m in all_affiliation_mismatches if not m['is_formatting_only']]

    discrepancies_by_doc = defaultdict(lambda: {
        'missed_authors': [],
        'extra_authors': [],
        'affiliation_mismatches': [],
        'formatting_only_mismatches': [],
        'real_mismatches': []
    })

    for item in all_missed_authors:
        discrepancies_by_doc[item['arxiv_id']]['missed_authors'].append({
            'name': item['author_name'],
            'name_normalized': item['author_name_normalized'],
            'affiliations': item['affiliations']
        })

    for item in all_extra_authors:
        discrepancies_by_doc[item['arxiv_id']]['extra_authors'].append({
            'name': item['author_name'],
            'name_normalized': item['author_name_normalized'],
            'affiliations': item['affiliations']
        })

    for item in all_affiliation_mismatches:
        doc_entry = {
            'author': item['author'],
            'ground_truth': item['ground_truth'],
            'predicted': item['predicted'],
            'matched_after_norm': item['matched_after_norm'],
            'missed_after_norm': item['missed_after_norm'],
            'extra_after_norm': item['extra_after_norm']
        }

        discrepancies_by_doc[item['arxiv_id']
                             ]['affiliation_mismatches'].append(doc_entry)

        if item['is_formatting_only']:
            discrepancies_by_doc[item['arxiv_id']
                                 ]['formatting_only_mismatches'].append(doc_entry)
        else:
            discrepancies_by_doc[item['arxiv_id']
                                 ]['real_mismatches'].append(doc_entry)

    return {
        'summary_statistics': summary_stats,
        'missed_authors': all_missed_authors,
        'extra_authors': all_extra_authors,
        'affiliation_mismatches': all_affiliation_mismatches,
        'formatting_only_mismatches': formatting_only_mismatches,
        'real_mismatches': real_mismatches,
        'discrepancies_by_document': dict(discrepancies_by_doc),
        'common_error_patterns': {
            'most_commonly_missed_affiliations_normalized': [
                {
                    'affiliation_normalized': norm,
                    'count': count,
                    # Show up to 3 variations
                    'original_variations': norm_to_original_missed[norm][:3]
                }
                for norm, count in common_missed_normalized
            ],
            'most_commonly_extra_affiliations_normalized': [
                {
                    'affiliation_normalized': norm,
                    'count': count,
                    'original_variations': norm_to_original_extra[norm][:3]
                }
                for norm, count in common_extra_normalized
            ]
        }
    }


def save_normalized_discrepancy_reports(analysis: Dict[str, Any], output_prefix: str):
    with open(f'{output_prefix}_complete.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Complete analysis saved to: {output_prefix}_complete.json")

    with open(f'{output_prefix}_summary.json', 'w') as f:
        json.dump(analysis['summary_statistics'], f, indent=2)
    print(f"Summary statistics saved to: {output_prefix}_summary.json")

    with open(f'{output_prefix}_real_mismatches.json', 'w') as f:
        json.dump({
            'summary': {
                'total_real_mismatches': len(analysis['real_mismatches']),
                'total_formatting_only': len(analysis['formatting_only_mismatches'])
            },
            'real_mismatches': analysis['real_mismatches']
        }, f, indent=2)
    print(f"Real mismatches saved to: {output_prefix}_real_mismatches.json")

    with open(f'{output_prefix}_error_patterns.json', 'w') as f:
        json.dump(analysis['common_error_patterns'], f, indent=2)
    print(f"Error patterns saved to: {output_prefix}_error_patterns.json")


def print_normalized_analysis_summary(analysis: Dict[str, Any]):
    stats = analysis['summary_statistics']

    print("\n" + "=" * 80)
    print("NORMALIZED DISCREPANCY ANALYSIS SUMMARY")
    print("=" * 80)

    print("\nOVERALL STATISTICS:")
    print("-" * 40)
    print(f"Total documents analyzed: {stats['total_documents']}")
    print(f"Documents with perfect author identification: {stats['documents_with_perfect_authors']} ({stats['documents_with_perfect_authors']/stats['total_documents']*100:.1f}%)")
    print(f"Documents with perfect affiliation extraction: {stats['documents_with_perfect_affiliations']} ({stats['documents_with_perfect_affiliations']/stats['total_documents']*100:.1f}%)")
    print(f"Documents with both perfect: {stats['documents_with_both_perfect']} ({stats['documents_with_both_perfect']/stats['total_documents']*100:.1f}%)")

    print("\nAFFILIATION MISMATCH ANALYSIS:")
    print("-" * 40)
    print(f"Total affiliation mismatches: {stats['total_affiliation_mismatches']}")
    print(f"  - Formatting differences only: {stats['affiliation_mismatches_formatting_only']} ({stats['affiliation_mismatches_formatting_only']/max(1, stats['total_affiliation_mismatches'])*100:.1f}%)")
    print(f"  - Real content differences: {stats['affiliation_mismatches_real']} ({stats['affiliation_mismatches_real']/max(1, stats['total_affiliation_mismatches'])*100:.1f}%)")

    print("\nAUTHOR-LEVEL DISCREPANCIES:")
    print("-" * 40)
    print(f"Total missed authors: {stats['total_missed_authors']}")
    print(f"Unique missed author names (normalized): {stats['unique_missed_author_names']}")
    print(f"Total extra authors: {stats['total_extra_authors']}")
    print(f"Unique extra author names (normalized): {stats['unique_extra_author_names']}")

    patterns = analysis['common_error_patterns']

    if patterns['most_commonly_missed_affiliations_normalized']:
        print("\nMOST COMMONLY MISSED AFFILIATIONS (Top 5, normalized):")
        print("-" * 40)
        for i, item in enumerate(patterns['most_commonly_missed_affiliations_normalized'][:5], 1):
            print(f"\n{i}. ({item['count']}x) {item['affiliation_normalized'][:80]}...")
            if item['original_variations']:
                print("   Original variations:")
                for var in item['original_variations'][:2]:
                    print(f"   - {var[:80]}...")

    docs_with_real_issues = {}
    for doc_id, disc in analysis['discrepancies_by_document'].items():
        real_count = (len(disc['missed_authors']) +
                      len(disc['extra_authors']) +
                      len(disc['real_mismatches']))
        if real_count > 0:
            docs_with_real_issues[doc_id] = real_count

    if docs_with_real_issues:
        sorted_docs = sorted(docs_with_real_issues.items(),
                             key=lambda x: x[1], reverse=True)
        print("\nDOCUMENTS WITH MOST REAL DISCREPANCIES (Top 5):")
        print("-" * 40)
        for doc_id, count in sorted_docs[:5]:
            disc = analysis['discrepancies_by_document'][doc_id]
            print(f"\n{doc_id}: {count} real discrepancies")
            print(f"  - Missed authors: {len(disc['missed_authors'])}")
            print(f"  - Extra authors: {len(disc['extra_authors'])}")
            print(f"  - Real affiliation mismatches: {len(disc['real_mismatches'])}")
            print(f"  - Formatting-only mismatches: {len(disc['formatting_only_mismatches'])}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze discrepancies with normalized comparison')
    parser.add_argument('-e', '--evaluation',
                        default='evaluation_results_detailed.json',
                        help='Path to evaluation results JSON file')
    parser.add_argument('-o', '--output-prefix',
                        default='normalized_discrepancy_analysis',
                        help='Prefix for output JSON files')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed analysis summary')

    args = parser.parse_args()

    print(f"Analyzing discrepancies with normalization from: {args.evaluation}")
    analysis = analyze_discrepancies_normalized(args.evaluation)
    save_normalized_discrepancy_reports(analysis, args.output_prefix)
    print_normalized_analysis_summary(analysis)
    if args.verbose:
        print("\nDetailed normalized discrepancy reports have been saved.")
        print("Real mismatches (excluding formatting issues) are in: "
              f"{args.output_prefix}_real_mismatches.json")


if __name__ == '__main__':
    main()
