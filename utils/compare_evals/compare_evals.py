import json
import argparse
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two eval JSON reports and emit precision/recall/F-score deltas as JSON.")
    parser.add_argument("-b", "--baseline", type=Path, help="Path to baseline eval JSON file")
    parser.add_argument("-c", "--candidate", type=Path, help="Path to candidate eval JSON file")
    parser.add_argument("-o", "--output", type=Path, help="Optional path to write JSON output")
    return parser.parse_args()


def load_eval(path: Path) -> Dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def main():
    args = parse_args()
    baseline = load_eval(args.baseline)
    candidate = load_eval(args.candidate)

    metrics_to_compare = ("precision", "recall", "f1_score", "f05_score", "f15_score")
    baseline_reports = baseline.get("reports", {})
    candidate_reports = candidate.get("reports", {})

    comparison: Dict[str, Any] = {}

    all_modes = sorted(set(baseline_reports) | set(candidate_reports))
    for mode in all_modes:
        mode_result: Dict[str, Any] = {}
        base_mode = baseline_reports.get(mode, {})
        cand_mode = candidate_reports.get(mode, {})
        all_categories = sorted(set(base_mode) | set(cand_mode))

        for category in all_categories:
            base_metrics = base_mode.get(category, {})
            cand_metrics = cand_mode.get(category, {})

            category_result: Dict[str, Any] = {}
            for metric in metrics_to_compare:
                base_value = base_metrics.get(metric)
                cand_value = cand_metrics.get(metric)
                if base_value is None or cand_value is None:
                    continue

                category_result[metric] = {
                    "baseline": base_value,
                    "candidate": cand_value,
                    "delta": cand_value - base_value,
                }

            if not category_result:
                continue

            if "f1_score" in category_result:
                category_result["delta_f1"] = category_result["f1_score"]["delta"]
            if "f05_score" in category_result:
                category_result["delta_f05"] = category_result["f05_score"]["delta"]
            if "f15_score" in category_result:
                category_result["delta_f15"] = category_result["f15_score"]["delta"]

            mode_result[category] = category_result

        if mode_result:
            comparison[mode] = mode_result

    output = {
        "baseline_path": str(args.baseline),
        "candidate_path": str(args.candidate),
        "metrics": comparison,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, sort_keys=True)
            fh.write("\n")
    else:
        print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
