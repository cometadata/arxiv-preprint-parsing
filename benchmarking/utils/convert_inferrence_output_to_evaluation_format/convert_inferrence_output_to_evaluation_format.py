import json
import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, TypedDict, Optional, Tuple


class Author(TypedDict):
    name: str
    affiliations: List[str]


def normalize_arxiv_id(arxiv_id: str) -> str:
    arxiv_id = arxiv_id.strip()
    has_prefix = arxiv_id.lower().startswith('arxiv:')
    if has_prefix:
        arxiv_id = arxiv_id[6:].strip()

    if '_' in arxiv_id and '/' not in arxiv_id:
        prefix, suffix = arxiv_id.split('_', 1)
        if prefix and suffix:
            arxiv_id = f"{prefix}/{suffix}"

    match = re.search(r'(\d{4}\.\d{4,5}(v\d+)?|[a-z\-]+/\d{7}(v\d+)?)', arxiv_id, re.IGNORECASE)
    if match:
        clean_id = match.group(1)
        return f"arXiv:{clean_id}"

    return f"arXiv:{arxiv_id}"


def process_predicted_authors(
    predicted_authors: Any,
    line_num: int = -1,
    context: Optional[str] = None,
) -> List[Author]:
    location = context or (f"Line {line_num}" if line_num != -1 else None)

    if predicted_authors is None:
        return []

    if not isinstance(predicted_authors, list):
        if location:
            logging.warning(
                f"{location}: 'predicted_authors' is not a list, but type {type(predicted_authors).__name__}"
            )
        return []

    processed_authors: List[Author] = []
    for i, author in enumerate(predicted_authors):
        if not isinstance(author, dict):
            if location:
                logging.warning(f"{location}: Author entry {i} is not a dict, skipping")
            continue

        if 'name' not in author or not author.get('name'):
            if location:
                logging.warning(
                    f"{location}: Author entry {i} is missing or has empty 'name', skipping"
                )
            continue

        affiliations = author.get('affiliations', [])
        if isinstance(affiliations, str):
            affiliations = [affiliations]
        elif not isinstance(affiliations, list):
            if location:
                logging.warning(
                    f"{location}: Author entry {i} has invalid 'affiliations' type {type(affiliations).__name__}, defaulting to empty list"
                )
            affiliations = []

        processed_authors.append({
            'name': author['name'],
            'affiliations': affiliations
        })

    return processed_authors


def _strip_known_suffixes(identifier: str) -> str:
    lowered = identifier.lower()
    for suffix in ('.tei.xml', '.pdf', '.jsonl', '.json', '.txt', '.xml'):
        if lowered.endswith(suffix):
            return identifier[: -len(suffix)]
    return identifier


def _extract_identifier(entry: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    priority_keys = [
        'arxiv_id',
        'arxivId',
        'paper_id',
        'doc_id',
        'document_id',
        'id',
        'filename',
        'file_name',
    ]

    for key in priority_keys:
        if key not in entry:
            continue

        value = entry.get(key)
        if value in (None, ""):
            continue

        raw_value = str(value).strip()
        if not raw_value:
            continue

        if key in {'filename', 'file_name'}:
            path = Path(raw_value)
            name = path.name
            name = _strip_known_suffixes(name)
            raw_value = name

        return raw_value, key

    return None, None


def _extract_prediction_list(entry: Dict[str, Any]) -> Any:
    for key in ('predicted_authors', 'prediction', 'predictions', 'authors'):
        if key in entry:
            return entry[key]
    return entry.get('predicted')


def _record_prediction(
    original_arxiv_id: Any,
    raw_predicted_authors: Any,
    predictions: Dict[str, Any],
    stats: Dict[str, int],
    gt_arxiv_ids: Dict[str, str],
    *,
    line_num: int = -1,
    context: Optional[str] = None,
    preserve_metadata: bool = False,
    entry: Optional[Dict[str, Any]] = None,
) -> None:
    arxiv_id = ''
    if original_arxiv_id is not None:
        arxiv_id = str(original_arxiv_id).strip()

    location = context or (f"Line {line_num}" if line_num != -1 else None)

    if not arxiv_id:
        if location:
            logging.error(f"{location}: missing arxiv_id")
        else:
            logging.error("Missing arxiv_id")
        stats['errors'] += 1
        return

    predicted_authors = process_predicted_authors(
        raw_predicted_authors,
        line_num=line_num,
        context=context,
    )

    if predicted_authors:
        stats['with_predictions'] += 1
    else:
        stats['null_predictions'] += 1

    normalized_id = normalize_arxiv_id(arxiv_id)
    target_id = gt_arxiv_ids.get(normalized_id, normalized_id)

    predictions[target_id] = {
        'predicted_authors': predicted_authors
    }

    if preserve_metadata and entry:
        for field in ['doi', 'title', 'filename', 'processing_time', 'error']:
            if field in entry and entry[field] is not None:
                predictions[target_id][field] = entry[field]


def convert_content_to_predictions(
    input_file: str,
    output_file: str,
    ground_truth_file: str = None,
    preserve_metadata: bool = False,
    wrap_predictions: bool = True
) -> None:
    predictions: Dict[str, Any] = {}
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

    parsed_data: Any = None
    parsed_successfully = False
    try:
        with open(input_file, 'r') as f:
            parsed_data = json.load(f)
            parsed_successfully = True
    except json.JSONDecodeError:
        parsed_data = None
    except Exception as exc:
        logging.debug(
            f"Unable to parse {input_file} as a single JSON document: {exc}"
        )
        parsed_data = None

    if parsed_successfully:
        if isinstance(parsed_data, dict):
            for idx, (raw_id, authors) in enumerate(parsed_data.items(), 1):
                stats['total'] += 1
                context = f"Entry {raw_id!r}"
                _record_prediction(
                    raw_id,
                    authors,
                    predictions,
                    stats,
                    gt_arxiv_ids,
                    context=context,
                    preserve_metadata=preserve_metadata,
                    entry={'arxiv_id': raw_id, 'predicted_authors': authors} if preserve_metadata else None,
                )
        elif isinstance(parsed_data, list):
            for idx, entry in enumerate(parsed_data, 1):
                if not isinstance(entry, dict):
                    logging.error(f"Entry {idx} is not a dict, skipping")
                    stats['errors'] += 1
                    continue

                stats['total'] += 1
                identifier, source_key = _extract_identifier(entry)
                if not identifier:
                    logging.error(f"Entry {idx} missing identifiable arxiv id fields")
                    stats['errors'] += 1
                    continue

                predicted_block = _extract_prediction_list(entry)
                context = f"Entry {idx}{f' ({source_key})' if source_key else ''}"
                _record_prediction(
                    identifier,
                    predicted_block,
                    predictions,
                    stats,
                    gt_arxiv_ids,
                    context=context,
                    preserve_metadata=preserve_metadata,
                    entry=entry if preserve_metadata else None,
                )
        else:
            logging.error(
                f"Unsupported JSON structure in {input_file}: {type(parsed_data).__name__}"
            )
            stats['errors'] += 1
    else:
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line)
                    if not isinstance(entry, dict):
                        logging.error(f"Line {line_num}: entry is not a dict, skipping")
                        stats['errors'] += 1
                        continue

                    stats['total'] += 1
                    identifier, source_key = _extract_identifier(entry)
                    if not identifier:
                        logging.error(f"Line {line_num}: missing identifiable arxiv id fields")
                        stats['errors'] += 1
                        continue

                    predicted_block = _extract_prediction_list(entry)
                    _record_prediction(
                        identifier,
                        predicted_block,
                        predictions,
                        stats,
                        gt_arxiv_ids,
                        line_num=line_num,
                        context=f"Line {line_num}{f' ({source_key})' if source_key else ''}",
                        preserve_metadata=preserve_metadata,
                        entry=entry if preserve_metadata else None,
                    )
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
    parser.add_argument(
        '-m', '--preserve-metadata',
        action='store_true',
        help='Preserve additional metadata fields (doi, title, filename, etc.) in output'
    )
    parser.add_argument(
        '--no-wrap',
        action='store_true',
        help='Do not wrap predictions in {"predicted_authors": [...]} format (legacy mode)'
    )

    args = parser.parse_args()

    if not args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)

    if not Path(args.input).exists():
        logging.error(f"Input file '{args.input}' not found")
        return 1

    convert_content_to_predictions(
        args.input,
        args.output,
        args.ground_truth,
        preserve_metadata=args.preserve_metadata,
        wrap_predictions=not args.no_wrap
    )
    return 0


if __name__ == '__main__':
    exit(main())
