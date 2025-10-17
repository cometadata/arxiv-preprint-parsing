import os
import sys
import json
import argparse
import tempfile
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Optional


def run_modal_volume_get(modal_cmd: str, volume: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [modal_cmd, "volume", "get", "--force", volume, remote_path, str(local_path)]
    subprocess.run(cmd, check=True)


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def merge_documents(pages: Iterable[Dict]) -> List[Dict]:
    grouped: Dict[str, Dict] = {}
    page_buckets: Dict[str, List[Dict]] = defaultdict(list)

    for record in pages:
        doc_id = record.get("document_id") or record.get("metadata", {}).get("Source-File")
        if doc_id is None:
            raise ValueError("Could not determine document_id for page record")

        doc = grouped.get(doc_id)
        if doc is None:
            doc = {
                "document_id": doc_id,
                "pdf_path": record.get("pdf_path"),
                "model": record.get("model"),
                "pages": [],
                "rotation_applied": {},
                "total_latency": 0.0,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "all_success": True,
            }
            grouped[doc_id] = doc

        page_buckets[doc_id].append(record)

        doc["total_latency"] += float(record.get("latency", 0.0))
        usage = record.get("usage") or {}
        doc["usage"]["prompt_tokens"] += usage.get("prompt_tokens", 0)
        doc["usage"]["completion_tokens"] += usage.get("completion_tokens", 0)
        doc["usage"]["total_tokens"] += usage.get("total_tokens") or (
            usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        )
        doc["rotation_applied"][record.get("page")] = record.get("rotation_applied", 0)
        doc["all_success"] = doc["all_success"] and record.get("success", False)

    documents: List[Dict] = []
    for doc_id, pages_for_doc in page_buckets.items():
        doc = grouped[doc_id]
        combined_text_parts: List[str] = []
        serialized_pages: List[Dict] = []

        for page_record in sorted(pages_for_doc, key=lambda r: (r.get("page_index", 0), r.get("page", 0))):
            content = (page_record.get("content") or "").rstrip()
            if content:
                combined_text_parts.append(content)

            serialized_pages.append(page_record)

        documents.append(
            {
                "document_id": doc_id,
                "pdf_path": doc.get("pdf_path"),
                "model": doc.get("model"),
                "success": doc["all_success"],
                "page_count": len(serialized_pages),
                "total_latency": doc["total_latency"],
                "usage": doc["usage"],
                "rotation_applied": doc["rotation_applied"],
                "text": "\n".join(part for part in combined_text_parts if part),
                "pages": serialized_pages,
            }
        )

    documents.sort(key=lambda d: d["document_id"])
    return documents


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default="rolmocr-data", help="Modal volume name")
    parser.add_argument(
        "--remote-predictions",
        default="rolmocr_predictions.jsonl",
        help="Path to the page-level predictions file on the Modal volume",
    )
    parser.add_argument(
        "--remote-errors",
        default=None,
        help="Optional path to the error JSONL to download alongside predictions",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write the merged document-level JSONL",
    )
    parser.add_argument(
        "--download-dir",
        default=".modal_downloads",
        help="Directory to store downloaded files (defaults to .modal_downloads)",
    )
    parser.add_argument(
        "--modal-cmd",
        default=os.environ.get("MODAL_CLI", "modal"),
        help="Modal CLI executable to use (default: 'modal' or $MODAL_CLI)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume local copies exist in download_dir and skip fetching from Modal",
    )
    args = parser.parse_args(argv)

    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    predictions_local = download_dir / Path(args.remote_predictions).name
    if not args.skip_download:
        run_modal_volume_get(args.modal_cmd, args.volume, args.remote_predictions, predictions_local)
        if args.remote_errors:
            run_modal_volume_get(args.modal_cmd, args.volume, args.remote_errors, download_dir / Path(args.remote_errors).name)

    page_records = list(iter_jsonl(predictions_local))
    if not page_records:
        print(f"No page records found in {predictions_local}", file=sys.stderr)
        return 1

    documents = merge_documents(page_records)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_f:
        for doc in documents:
            out_f.write(json.dumps(doc, ensure_ascii=False))
            out_f.write("\n")

    print(f"Wrote {len(documents)} documents to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
