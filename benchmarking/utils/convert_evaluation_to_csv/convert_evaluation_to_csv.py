import csv
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_evaluation(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def is_perfect_document(doc: Dict[str, Any]) -> bool:
    if doc.get("status") != "success":
        return False
    combined = doc.get("metrics", {}).get("combined", {})
    strict = combined.get("strict")
    if not strict:
        return False
    return strict.get("false_positives", 0) == 0 and strict.get("false_negatives", 0) == 0


def extract_missing_extra_authors(doc: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    missing: List[str] = []
    extra: List[str] = []
    for match in doc.get("details", {}).get("entity_matches", []):
        match_type = match.get("match_type")
        if match_type == "unmatched_ground_truth":
            gt = match.get("ground_truth") or {}
            name = gt.get("name")
            if name:
                missing.append(name)
        elif match_type == "unmatched_prediction":
            pred = match.get("prediction") or {}
            name = pred.get("name")
            if name:
                extra.append(name)
    return missing, extra


def summarize_field_issues(doc: Dict[str, Any]) -> Dict[str, List[str]]:
    details = doc.get("details", {})
    matches: List[Dict[str, Any]] = details.get("entity_matches", [])
    array_fields = details.get("array_field_details", {})

    if not matches or not array_fields:
        return {}

    indices = {field: 0 for field in array_fields}
    issues: Dict[str, List[str]] = {field: [] for field in array_fields}

    for match in matches:
        match_type = match.get("match_type")
        if match_type not in {"strict", "fuzzy"}:
            continue

        for field_name, entries in array_fields.items():
            idx = indices[field_name]
            if idx >= len(entries):
                continue
            entry = entries[idx]
            indices[field_name] = idx + 1

            strict_info = entry.get("strict", {})
            missed = strict_info.get("missed") or []
            extra = strict_info.get("extra") or []
            if not missed and not extra:
                continue

            name = (match.get("ground_truth") or {}).get("name")
            if not name:
                name = (match.get("prediction") or {}).get("name")
            if not name:
                name = "UNKNOWN"

            parts: List[str] = []
            if missed:
                parts.append("missed=" + " | ".join(missed))
            if extra:
                parts.append("extra=" + " | ".join(extra))
            issues[field_name].append(f"{name}: {'; '.join(parts)}")

    return {field: vals for field, vals in issues.items() if vals}


def metrics_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    combined = metrics.get("combined", {}) if isinstance(metrics, dict) else {}
    strict = combined.get("strict") if isinstance(combined, dict) else None
    result = {
        "combined_strict_precision": format_float(strict.get("precision")) if strict else "",
        "combined_strict_recall": format_float(strict.get("recall")) if strict else "",
        "combined_strict_f1": format_float(strict.get("f1_score")) if strict else "",
        "combined_strict_false_positives": strict.get("false_positives") if strict else "",
        "combined_strict_false_negatives": strict.get("false_negatives") if strict else "",
    }

    entity = metrics.get("entity:author") if isinstance(metrics, dict) else None
    if isinstance(entity, dict):
        strict_entity = entity.get("strict")
        if strict_entity:
            total_gt = (strict_entity.get("true_positives", 0) or 0) + (strict_entity.get("false_negatives", 0) or 0)
            total_pred = (strict_entity.get("true_positives", 0) or 0) + (strict_entity.get("false_positives", 0) or 0)
            result.update(
                {
                    "entity_true_positives": strict_entity.get("true_positives"),
                    "entity_false_positives": strict_entity.get("false_positives"),
                    "entity_false_negatives": strict_entity.get("false_negatives"),
                    "ground_truth_authors": total_gt,
                    "predicted_authors": total_pred,
                }
            )
    return result


def generate_error_detail_rows(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    doc_id = doc.get("doc_id")
    status = doc.get("status")

    if status != "success":
        rows.append(
            {
                "doc_id": doc_id,
                "status": status,
                "match_level": "",
                "issue_type": "document_error",
                "entity_name": "",
                "field_name": "",
                "ground_truth": "",
                "prediction": "",
                "detail": doc.get("error", ""),
            }
        )
        return rows

    details = doc.get("details", {})
    matches: List[Dict[str, Any]] = details.get("entity_matches", [])
    array_fields = details.get("array_field_details", {})

    indices = {field: 0 for field in array_fields}

    for match in matches:
        match_type = match.get("match_type")
        ground = match.get("ground_truth") or {}
        pred = match.get("prediction") or {}
        entity_name = ground.get("name") or pred.get("name") or ""

        base = {
            "doc_id": doc_id,
            "status": status,
            "entity_name": entity_name,
        }

        if match_type == "unmatched_ground_truth":
            rows.append(
                {
                    **base,
                    "match_level": "strict",
                    "issue_type": "missing_author",
                    "field_name": "name",
                    "ground_truth": ground.get("name", ""),
                    "prediction": "",
                    "detail": " | ".join(ground.get("affiliations", [])),
                }
            )
            continue
        if match_type == "unmatched_prediction":
            rows.append(
                {
                    **base,
                    "match_level": "strict",
                    "issue_type": "extra_author",
                    "field_name": "name",
                    "ground_truth": "",
                    "prediction": pred.get("name", ""),
                    "detail": " | ".join(pred.get("affiliations", [])),
                }
            )
            continue

        if match_type == "fuzzy":
            rows.append(
                {
                    **base,
                    "match_level": "fuzzy",
                    "issue_type": "name_mismatch",
                    "field_name": "name",
                    "ground_truth": ground.get("name", ""),
                    "prediction": pred.get("name", ""),
                    "detail": f"similarity={match.get('similarity')}",
                }
            )

        if match_type not in {"strict", "fuzzy"}:
            continue

        for field_name, entries in array_fields.items():
            idx = indices[field_name]
            if idx >= len(entries):
                continue
            entry = entries[idx]
            indices[field_name] = idx + 1

            strict_info = entry.get("strict") or {}
            missed = strict_info.get("missed") or []
            extra = strict_info.get("extra") or []
            if missed or extra:
                rows.append(
                    {
                        **base,
                        "match_level": "strict",
                        "issue_type": f"{field_name}_mismatch",
                        "field_name": field_name,
                        "ground_truth": " | ".join(missed),
                        "prediction": " | ".join(extra),
                        "detail": "",
                    }
                )

            fuzzy_info = entry.get("fuzzy") or {}
            unmatched_gt = fuzzy_info.get("unmatched_gt_norms") or []
            unmatched_pred = fuzzy_info.get("unmatched_pred_norms") or []
            if unmatched_gt or unmatched_pred:
                rows.append(
                    {
                        **base,
                        "match_level": "fuzzy",
                        "issue_type": f"{field_name}_mismatch",
                        "field_name": field_name,
                        "ground_truth": " | ".join(unmatched_gt),
                        "prediction": " | ".join(unmatched_pred),
                        "detail": "",
                    }
                )

    return rows


def format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            filtered = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(filtered)


def export_csvs(evaluation: Dict[str, Any], correct_path: Path, error_path: Path) -> None:
    correct_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    for doc in evaluation.get("document_results", []):
        base = {
            "doc_id": doc.get("doc_id"),
            "status": doc.get("status"),
        }
        metrics_info = metrics_summary(doc.get("metrics", {}))
        row = {**base, **metrics_info}

        if is_perfect_document(doc):
            correct_rows.append(row)
            continue

        detail_rows.extend(generate_error_detail_rows(doc))
        missing, extra = extract_missing_extra_authors(doc)
        field_issues = summarize_field_issues(doc)
        notes: List[str] = []
        if missing:
            notes.append("missing_authors=" + " | ".join(missing))
        if extra:
            notes.append("extra_authors=" + " | ".join(extra))
        if doc.get("status") != "success" and doc.get("error"):
            notes.append(f"error={doc['error']}")
        for field, entries in field_issues.items():
            notes.append(f"{field}_issues=" + " || ".join(entries))

        row["notes"] = " ; ".join(notes)
        error_rows.append(row)

    correct_fields = [
        "doc_id",
        "status",
        "ground_truth_authors",
        "predicted_authors",
        "combined_strict_precision",
        "combined_strict_recall",
        "combined_strict_f1",
    ]

    error_fields = correct_fields + [
        "combined_strict_false_positives",
        "combined_strict_false_negatives",
        "entity_true_positives",
        "entity_false_positives",
        "entity_false_negatives",
        "notes",
    ]

    write_csv(correct_path, correct_fields, correct_rows)
    write_csv(error_path, error_fields, error_rows)
    if detail_rows:
        detail_fields = [
            "doc_id",
            "status",
            "match_level",
            "issue_type",
            "entity_name",
            "field_name",
            "ground_truth",
            "prediction",
            "detail",
        ]
        detail_path = error_path.with_name(error_path.stem + "_details" + error_path.suffix)
        write_csv(detail_path, detail_fields, detail_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export evaluation JSON into readable CSVs for correct and problematic documents",
    )
    parser.add_argument("-i", "--input", required=True, help="Path to evaluation JSON output")
    parser.add_argument(
        "-o",
        "--output-prefix",
        required=True,
        help="Prefix path for generated CSV files (e.g. results/run1)",
    )
    parser.add_argument(
        "--correct-suffix",
        default="_correct.csv",
        help="Suffix appended to prefix for correct predictions CSV",
    )
    parser.add_argument(
        "--error-suffix",
        default="_errors.csv",
        help="Suffix appended to prefix for error summaries CSV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input evaluation file not found: {input_path}")

    evaluation = load_evaluation(input_path)
    prefix = Path(args.output_prefix)
    correct_path = prefix.with_suffix("")
    if correct_path == prefix:
        correct_path = prefix
    correct_file = correct_path.parent / (correct_path.name + args.correct_suffix)
    error_file = correct_path.parent / (correct_path.name + args.error_suffix)

    export_csvs(evaluation, correct_file, error_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
