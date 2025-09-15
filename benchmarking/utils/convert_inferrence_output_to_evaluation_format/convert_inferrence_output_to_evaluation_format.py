import json
import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, TypedDict


class Author(TypedDict):
    name: str
    affiliations: List[str]


def normalize_arxiv_id(arxiv_id: str) -> str:
    match = re.search(r'(\d{4}\.\d{4,5}(v\d+)?|[a-z\-]+/\d{7}(v\d+)?)', arxiv_id, re.IGNORECASE)
    if match:
        return match.group(1)

    if arxiv_id.lower().startswith('arxiv:'):
        return arxiv_id[6:]

    return arxiv_id


def process_predicted_authors(predicted_authors: Any, line_num: int = -1) -> List[Author]:
    if predicted_authors is None:
        return []

    if not isinstance(predicted_authors, list):
        if line_num != -1:
            logging.warning(f"Line {line_num}: 'predicted_authors' is not a list, but type {type(predicted_authors).__name__}")
        return []

    processed_authors: List[Author] = []
    for i, author in enumerate(predicted_authors):
        if not isinstance(author, dict):
            if line_num != -1:
                logging.warning(f"Line {line_num}: Author entry {i} is not a dict, skipping")
            continue

        if 'name' not in author or not author.get('name'):
            if line_num != -1:
                logging.warning(f"Line {line_num}: Author entry {i} is missing or has empty 'name', skipping")
            continue

        affiliations = author.get('affiliations', [])
        if not isinstance(affiliations, list):
            affiliations = []

        processed_authors.append({
            'name': author['name'],
            'affiliations': affiliations
        })

    return processed_authors


def convert_content_to_predictions(input_file: str, output_file: str, ground_truth_file: str = None) -> None:
    predictions: Dict[str, List[Author]] = {}
    stats = {
        'total': 0,
        'with_predictions': 0,
        'null_predictions': 0,
        'errors': 0
    }

    gt_arxiv_ids: Dict[str, str] = {}
    if ground_truth_file and Path(ground_truth_file).exists():
        logging.info(f"Loading ground truth from {ground_truth_file} to match ID format...")
        with open(ground_truth_file, 'r') as f:
            gt_data = json.load(f)
            for doc in gt_data:
                norm_id = normalize_arxiv_id(doc['arxiv_id'])
                gt_arxiv_ids[norm_id] = doc['arxiv_id']

    logging.info(f"Reading from {input_file}...")

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                entry = json.loads(line)
                stats['total'] += 1

                original_arxiv_id = entry.get('arxiv_id', '')
                if not original_arxiv_id:
                    logging.error(f"Line {line_num} missing arxiv_id")
                    stats['errors'] += 1
                    continue

                predicted_authors = process_predicted_authors(entry.get('predicted_authors'), line_num=line_num)

                if predicted_authors:
                    stats['with_predictions'] += 1
                else:
                    stats['null_predictions'] += 1

                normalized_id = normalize_arxiv_id(original_arxiv_id)
                if normalized_id in gt_arxiv_ids:
                    predictions[gt_arxiv_ids[normalized_id]] = predicted_authors
                else:
                    predictions[original_arxiv_id] = predicted_authors

            except json.JSONDecodeError as e:
                logging.error(f"Error parsing JSON on line {line_num}: {e}")
                stats['errors'] += 1
            except Exception as e:
                logging.error(f"Error processing line {line_num}: {e}")
                stats['errors'] += 1

    logging.info(f"Writing to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    logging.info("=== Conversion Statistics ===")
    logging.info(f"Total entries processed: {stats['total']}")
    logging.info(f"Entries with predictions: {stats['with_predictions']}")
    logging.info(f"Entries with null/empty predictions: {stats['null_predictions']}")
    logging.info(f"Errors encountered: {stats['errors']}")
    logging.info(f"Output saved to: {output_file}")


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        stream=sys.stdout
    )

    parser = argparse.ArgumentParser(
        description='Convert content.jsonl to predictions format for evaluation script'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to input content.jsonl file (default: content.jsonl)'
    )
    parser.add_argument(
        '-o', '--output',
        default='predictions.json',
        help='Path to output predictions.json file (default: predictions.json)'
    )
    parser.add_argument(
        '-g', '--ground-truth',
        help='Path to ground truth JSON file to match arxiv_id format (optional)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (show warnings about skipped entries)'
    )

    args = parser.parse_args()

    if not args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)

    if not Path(args.input).exists():
        logging.error(f"Input file '{args.input}' not found")
        return 1

    convert_content_to_predictions(args.input, args.output, args.ground_truth)
    return 0


if __name__ == '__main__':
    exit(main())