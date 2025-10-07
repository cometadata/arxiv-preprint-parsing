import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional


def _run_modal_command(args: List[str], modal_env: Optional[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if modal_env:
        env["MODAL_ENVIRONMENT"] = modal_env
    return subprocess.run(["modal", *args], capture_output=True, text=True, env=env)


def _ensure_volume(volume: str, modal_env: Optional[str]) -> None:
    result = _run_modal_command(["volume", "create", volume], modal_env)
    if result.returncode != 0:
        output = result.stderr.strip() or result.stdout.strip()
        if "already exists" not in output.lower():
            raise RuntimeError(f"modal volume create failed for {volume}: {output}")


def _run_modal_volume_put(volume: str, local: Path, remote: str, modal_env: Optional[str]) -> None:
    result = _run_modal_command(["volume", "put", volume, str(local), remote], modal_env)
    if result.returncode != 0:
        raise RuntimeError(f"modal volume put failed for {local}: {result.stderr.strip() or result.stdout.strip()}")


def _collect_pdfs(pdf_dir: Path, recursive: bool) -> List[Path]:
    candidates: Iterable[Path]
    if recursive:
        candidates = pdf_dir.rglob("*.pdf")
    else:
        candidates = pdf_dir.glob("*.pdf")
    pdfs = sorted(path for path in candidates if path.is_file())
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")
    return pdfs


def _write_manifest(entries: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for entry in entries:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _upload_prompt(volume: str, prompt_path: Path, remote_prefix: str, modal_env: Optional[str]) -> str:
    remote_rel = f"/{remote_prefix.strip('/')}/{prompt_path.name}"
    _run_modal_volume_put(volume, prompt_path, remote_rel, modal_env)
    return f"/data{remote_rel}"


def prepare_inputs(
    volume: str,
    pdf_dir: Path,
    manifest_local: Path,
    manifest_remote: str,
    pdf_prefix: str,
    recursive: bool,
    prompt_path: Optional[Path],
    prompt_prefix: str,
    modal_env: Optional[str],
) -> dict:
    _ensure_volume(volume, modal_env)
    pdfs = _collect_pdfs(pdf_dir, recursive)

    manifest_remote_rel = f"/{manifest_remote.lstrip('/')}"

    manifest_entries: List[dict] = []
    for pdf_path in pdfs:
        remote_rel = f"/{pdf_prefix.strip('/')}/{pdf_path.name}"
        _run_modal_volume_put(volume, pdf_path, remote_rel, modal_env)
        manifest_entries.append(
            {
                "document_id": pdf_path.stem,
                "pdf_path": f"/data{remote_rel}",
            }
        )

    _write_manifest(manifest_entries, manifest_local)
    _run_modal_volume_put(volume, manifest_local, manifest_remote_rel, modal_env)

    prompt_remote = None
    if prompt_path is not None:
        prompt_remote = _upload_prompt(volume, prompt_path, prompt_prefix, modal_env)

    return {
        "pdf_count": len(manifest_entries),
        "manifest_local": str(manifest_local),
        "manifest_remote": f"/data{manifest_remote_rel}",
        "prompt_remote": prompt_remote,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Upload PDFs to Modal and build RolmOCR manifest")
    parser.add_argument("pdf_dir", type=Path, help="Directory containing PDF files")
    parser.add_argument("manifest", type=Path, help="Local path for the generated manifest")
    parser.add_argument(
        "--volume",
        default="rolmocr-data",
        help="Modal volume name (default: rolmocr-data)",
    )
    parser.add_argument(
        "--manifest-remote",
        default="manifest.jsonl",
        help="Remote path inside the Modal volume for the manifest (default: manifest.jsonl)",
    )
    parser.add_argument(
        "--pdf-prefix",
        default="pdfs",
        help="Remote prefix inside the Modal volume for uploaded PDFs (default: pdfs)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively upload PDFs from subdirectories",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        help="Optional prompt file to upload alongside the PDFs",
    )
    parser.add_argument(
        "--prompt-prefix",
        default="prompts",
        help="Remote prefix for the prompt file when --prompt is provided",
    )
    parser.add_argument(
        "--modal-env",
        help="Modal environment name (e.g. main, staging). If omitted, uses default",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.pdf_dir.exists() or not args.pdf_dir.is_dir():
        raise NotADirectoryError(f"PDF directory not found: {args.pdf_dir}")
    if args.prompt is not None and not args.prompt.is_file():
        raise FileNotFoundError(f"Prompt file not found: {args.prompt}")

    result = prepare_inputs(
        volume=args.volume,
        pdf_dir=args.pdf_dir,
        manifest_local=args.manifest,
        manifest_remote=args.manifest_remote,
        pdf_prefix=args.pdf_prefix,
        recursive=args.recursive,
        prompt_path=args.prompt,
        prompt_prefix=args.prompt_prefix,
        modal_env=args.modal_env,
    )

    print(json.dumps(result, indent=2))
    prompt_flag = ""
    if result.get("prompt_remote"):
        prompt_flag = f" --system-prompt {result['prompt_remote']}"

    print(
        "Run the batch with: modal run benchmarking_lora/rolmocr_batch_inference_vllm_modal.py "
        f"--manifest /data/{args.manifest_remote}{prompt_flag}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
