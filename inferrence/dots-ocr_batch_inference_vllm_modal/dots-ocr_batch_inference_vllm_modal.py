import os
import re
import sys
import ssl
import time
import json
import base64
import random
import shutil
import asyncio
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from io import BytesIO
from urllib.parse import urlparse

import modal

try:
    import boto3
except ImportError:
    boto3 = None  # type: ignore

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # type: ignore


app = modal.App("rolmocr-vllm-batch")


def _build_image() -> modal.Image:
    base = modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    return (
        base.apt_install(
            "git",
            "wget",
            "poppler-utils",
            "libgl1",
            "libglib2.0-0",
        )
        .pip_install(
            "torch==2.7.1",
            "transformers>=4.44.0",
            "huggingface_hub[hf_transfer]>=0.30.0",
            "openai>=1.25.0",
            "httpx",
            "tqdm",
            "boto3",
            "numpy<2",
            "pypdf",
            "Pillow",
            "pymupdf",
            "cryptography>=43.0.3",
            extra_index_url="https://wheels.vllm.ai/nightly",
        )
        .pip_install(
            "vllm>=0.9.1",
            extra_index_url="https://wheels.vllm.ai/nightly",
        )
        .env(
            {
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "VLLM_USE_V1": "1",
                "OMP_NUM_THREADS": "1",
            }
        )
    )


DATA_VOLUME = modal.Volume.from_name("rolmocr-data", create_if_missing=True)
MODEL_VOLUME = modal.Volume.from_name("dots-ocr-models", create_if_missing=True)
VLLM_CACHE_VOLUME = modal.Volume.from_name("dots-ocr-vllm-cache", create_if_missing=True)

IMAGE = _build_image()

DEFAULT_MODEL = os.environ.get("DOTS_MODEL", "rednote-hilab/dots.ocr")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "30024"))
MODEL_MAX_CONTEXT = 131072
MAX_COMPLETION_TOKENS = 65536

_DEFAULT_RENDER_WORKERS = max(1, (os.cpu_count() or 4) - 1)
_RENDER_WORKER_CAP = max(1, int(os.environ.get("DOTS_RENDER_WORKERS", _DEFAULT_RENDER_WORKERS)))
PDF_RENDER_MAX_WORKERS = asyncio.BoundedSemaphore(_RENDER_WORKER_CAP)


PROMPT_TEMPLATES = {
    "ocr": """Convert this document image to markdown format. Extract all text while preserving the layout and structure. Use proper markdown formatting for headings, lists, tables, and other elements.""",

    "layout-all": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.""",

    "layout-only": """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.""",
}


@dataclass
class BatchConfig:
    input_manifest: Path = Path("/data/manifest.jsonl")
    output_path: Path = Path("/data/dots_predictions.jsonl")
    errored_path: Path = Path("/data/dots_errors.jsonl")
    download_dir: Path = Path("/tmp/pdfs")
    max_pages_per_doc: Optional[int] = None
    workers: int = 4
    max_concurrent_docs: int = 8
    max_inflight_requests: int = 128
    request_timeout: float = 180.0
    max_retries: int = 3
    initial_temperature: float = 0.0
    temperature_increment: float = 0.1
    target_longest_dim: int = 1024
    batch_id: Optional[str] = None
    prompt_mode: str = "layout-all"
    target_pdf_filenames: Optional[Set[str]] = None


@dataclass
class ManifestEntry:
    document_id: str
    pdf_path: str
    pages: Optional[List[int]] = None

    @staticmethod
    def from_jsonl(line: str) -> "ManifestEntry":
        record = json.loads(line)
        return ManifestEntry(
            document_id=record.get("document_id") or record.get("id"),
            pdf_path=record["pdf_path"],
            pages=record.get("pages"),
        )


class Metrics:
    def __init__(self) -> None:
        self.total_pages = 0
        self.successful_pages = 0
        self.failed_pages = 0
        self.total_latency = 0.0

    def record(self, success: bool, latency: float) -> None:
        self.total_pages += 1
        if success:
            self.successful_pages += 1
        else:
            self.failed_pages += 1
        self.total_latency += latency

    def as_dict(self) -> Dict[str, Any]:
        avg_latency = self.total_latency / self.total_pages if self.total_pages else 0.0
        return {
            "total_pages": self.total_pages,
            "successful_pages": self.successful_pages,
            "failed_pages": self.failed_pages,
            "average_latency": avg_latency,
        }


def setup_logging(log_path: Path = Path("/data/dots_batch.log")) -> logging.Logger:
    logger = logging.getLogger("dots-batch")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def parse_s3_path(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {s3_uri}")
    without_scheme = s3_uri[len("s3://") :]
    bucket, _, key = without_scheme.partition("/")
    if not bucket or not key:
        raise ValueError(f"Malformed s3 uri: {s3_uri}")
    return bucket, key


def ensure_pdf_local(pdf_path: str, download_dir: Path, logger: logging.Logger) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)

    if pdf_path.startswith("s3://"):
        if boto3 is None:
            raise RuntimeError("boto3 is required to download from S3 but is not installed")
        bucket, key = parse_s3_path(pdf_path)
        sanitized_name = key.replace("/", "_")
        local_path = download_dir / sanitized_name
        if not local_path.exists():
            logger.info(f"Downloading {pdf_path} -> {local_path}")
            s3 = boto3.client("s3")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_path))
        return local_path

    candidate = Path(pdf_path)
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"PDF path not found: {pdf_path}")


def extract_pdf_filename(pdf_path: str) -> str:
    """Return the filename component for local paths and URLs/S3 URIs."""
    parsed = urlparse(pdf_path)
    if parsed.scheme and parsed.path:
        return Path(parsed.path).name
    return Path(pdf_path).name


def is_png(file_path: str) -> bool:
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
            return header[:8] == b"\x89PNG\r\n\x1a\n"
    except Exception:
        return False


def is_jpeg(file_path: str) -> bool:
    try:
        with open(file_path, "rb") as f:
            header = f.read(3)
            return header[:3] == b"\xff\xd8\xff"
    except Exception:
        return False


def render_pdf_to_base64png(pdf_path: str, page_num: int, target_longest_dim: int = 1024) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required but not installed")
    if Image is None:
        raise RuntimeError("Pillow is required but not installed")

    doc = fitz.open(pdf_path)

    try:
        page = doc[page_num - 1]

        page_rect = page.rect
        current_longest = max(page_rect.width, page_rect.height)
        zoom = target_longest_dim / current_longest

        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return b64_string

    finally:
        doc.close()


async def wait_for_vllm(base_url: str, timeout: int = 600) -> None:
    if httpx is None:
        raise RuntimeError("httpx must be available to wait for the server")

    deadline = time.time() + timeout
    url = f"{base_url.rstrip('/')}/v1/models"

    async with httpx.AsyncClient(timeout=30) as client:
        while time.time() < deadline:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(2)

    raise TimeoutError("vLLM server failed to start within timeout")


def launch_vllm_server(
    model_path: Path,
    tensor_parallel_size: int,
    gpu_memory_utilization: Optional[float],
    max_model_len: int,
    semaphore: asyncio.Semaphore,
    max_doc_concurrency: int,
    logger: logging.Logger,
) -> tuple[subprocess.Popen[str], asyncio.Task[Any]]:
    cache_dir = Path("/root/.cache/vllm")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        "vllm",
        "serve",
        str(model_path),
        "--port",
        str(VLLM_PORT),
        "--served-model-name",
        "dots-ocr",
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--disable-log-requests",
        "--uvicorn-log-level",
        "warning",
        "--trust-remote-code",
    ]

    if gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])

    env = {**os.environ, "VLLM_USE_V1": "1", "VLLM_CACHE_DIR": str(cache_dir)}
    logger.info("Starting vLLM server: %s", " ".join(cmd))
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    monitor_task = asyncio.create_task(monitor_vllm_logs(process, semaphore, max_doc_concurrency, logger))
    return process, monitor_task


async def monitor_vllm_logs(
    process: subprocess.Popen[str],
    semaphore: asyncio.Semaphore,
    max_doc_concurrency: int,
    logger: logging.Logger,
) -> None:
    assert process.stdout and process.stderr

    last_running_req = 0
    peak_running_req = 0
    last_queue_req = 0
    running_reqs_decreased = False
    server_ready = False
    last_semaphore_release = time.time()
    current_slots = 1

    async def process_line(line: str) -> None:
        nonlocal last_running_req, peak_running_req, last_queue_req, running_reqs_decreased, server_ready, last_semaphore_release, current_slots
        if not line:
            return
        logger.info(line)

        if not server_ready and ("The server is fired up and ready to roll!" in line or "Starting vLLM API server" in line):
            server_ready = True
            last_semaphore_release = time.time()

        if "Detected errors during sampling" in line:
            logger.error("vLLM reported sampling errors; exiting")
            raise RuntimeError("vLLM sampling error")

        if match := re.search(r"Running: (\d+)", line):
            current_running = int(match.group(1))
            if current_running > peak_running_req:
                peak_running_req = current_running
                logger.info(f"New peak running requests: {peak_running_req}")
            if current_running < last_running_req and not running_reqs_decreased:
                running_reqs_decreased = True
                logger.info(f"Running requests decreased: {last_running_req} -> {current_running}")
            last_running_req = current_running

        if match := re.search(r"(?:Waiting|Pending):\s*(\d+)", line):
            last_queue_req = int(match.group(1))
            logger.info(f"vLLM running req: {last_running_req} queue req: {last_queue_req}")

        idle_time = time.time() - last_semaphore_release
        low_queue = last_queue_req <= max(1, peak_running_req // 10)
        should_release = (
            server_ready
            and semaphore.locked()
            and current_slots < max_doc_concurrency
            and idle_time > 10
            and (low_queue or last_running_req == 0 or running_reqs_decreased)
        )

        if should_release:
            semaphore.release()
            current_slots += 1
            running_reqs_decreased = False
            last_semaphore_release = time.time()
            logger.info(
                "Semaphore released (slots=%s, running=%s, queued=%s, peak=%s)",
                current_slots,
                last_running_req,
                last_queue_req,
                peak_running_req,
            )

    async def read_stream(stream, level: int) -> None:
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, stream.readline)
            if not line:
                break
            try:
                await process_line(line.rstrip())
            except Exception as exc:
                logger.error("Log monitor error: %s", exc)

    stdout_task = asyncio.create_task(read_stream(process.stdout, logging.INFO))
    stderr_task = asyncio.create_task(read_stream(process.stderr, logging.ERROR))

    try:
        await asyncio.to_thread(process.wait)
    finally:
        stdout_task.cancel()
        stderr_task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)


def build_payload(
    image_b64: str,
    temperature: float,
    model_name: str,
    prompt_text: str,
) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []

    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        },
        {
            "type": "text",
            "text": prompt_text,
        },
    ]

    messages.append({"role": "user", "content": user_content})

    return {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "top_p": 0.95,
    }


def temperature_for_attempt(cfg: BatchConfig, attempt: int) -> float:
    return cfg.initial_temperature + attempt * cfg.temperature_increment


async def apost(url: str, json_data: Dict[str, Any], api_key: Optional[str] = None) -> tuple[int, bytes]:
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    if host is None:
        raise ValueError(f"Invalid URL: {url}")

    if parsed_url.scheme == "https":
        port = parsed_url.port or 443
        use_ssl = True
    else:
        port = parsed_url.port or 80
        use_ssl = False

    path = parsed_url.path or "/"
    writer = None

    try:
        if use_ssl:
            ssl_context = ssl.create_default_context()
            reader, writer = await asyncio.open_connection(host, port, ssl=ssl_context)
        else:
            reader, writer = await asyncio.open_connection(host, port)

        json_payload = json.dumps(json_data)

        headers = [
            f"POST {path} HTTP/1.1",
            f"Host: {host}",
            "Content-Type: application/json",
            f"Content-Length: {len(json_payload)}",
        ]

        if api_key:
            headers.append(f"Authorization: Bearer {api_key}")

        headers.append("Connection: close")

        request = "\r\n".join(headers) + "\r\n\r\n" + json_payload
        writer.write(request.encode())
        await writer.drain()

        status_line = await reader.readline()
        if not status_line:
            raise ConnectionError("No response from server")
        status_parts = status_line.decode().strip().split(" ", 2)
        if len(status_parts) < 2:
            raise ValueError(f"Malformed status line: {status_line.decode().strip()}")
        status_code = int(status_parts[1])

        response_headers: Dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            key, _, value = line.decode().partition(":")
            response_headers[key.strip().lower()] = value.strip()

        if "content-length" in response_headers:
            body_length = int(response_headers["content-length"])
            response_body = await reader.readexactly(body_length)
        else:
            raise ConnectionError("Unsupported response without Content-Length header")

        return status_code, response_body
    finally:
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


async def process_page(
    *,
    entry: ManifestEntry,
    cfg: BatchConfig,
    model_name: str,
    logger: logging.Logger,
    page_num: int,
    page_index: int,
    local_pdf: Path,
    completion_url: str,
    api_key: Optional[str],
    request_semaphore: asyncio.Semaphore,
    prompt_text: str,
) -> Dict[str, Any]:
    start = time.time()
    output_text: Optional[str] = None
    usage: Dict[str, Any] = {}
    error: Optional[str] = None
    success = False

    try:
        async with PDF_RENDER_MAX_WORKERS:
            base_image_b64 = await asyncio.to_thread(
                render_pdf_to_base64png,
                str(local_pdf),
                page_num,
                cfg.target_longest_dim,
            )
    except Exception as exc:
        error = f"render_failed: {exc}"
        latency = time.time() - start
        return {
            "document_id": entry.document_id,
            "page": page_num,
            "page_index": page_index,
            "pdf_path": entry.pdf_path,
            "model": model_name,
            "success": success,
            "latency": latency,
            "usage": usage,
            "content": output_text,
            "error": error,
        }

    attempt = 0
    exponential_backoffs = 0

    while attempt < cfg.max_retries and not success:
        payload = build_payload(
            image_b64=base_image_b64,
            temperature=temperature_for_attempt(cfg, attempt),
            model_name=model_name,
            prompt_text=prompt_text,
        )

        try:
            async with request_semaphore:
                status_code, response_body = await apost(completion_url, payload, api_key=api_key)
            body_text = response_body.decode("utf-8", errors="replace")

            if status_code == 429:
                raise ConnectionError("Too many requests (429)")
            if status_code >= 500:
                raise ConnectionError(f"Server error {status_code}")
            if status_code == 400:
                raise ValueError(f"Bad request: {body_text}")
            if status_code != 200:
                raise ValueError(f"HTTP {status_code}: {body_text}")

            base_response_data = json.loads(body_text)
            usage = base_response_data.get("usage", {})

            if usage.get("total_tokens", 0) > MODEL_MAX_CONTEXT:
                raise ValueError(f"Response exceeded model_max_context of {MODEL_MAX_CONTEXT}")

            choice = base_response_data["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError(f"Unexpected finish_reason: {choice.get('finish_reason')}")

            output_text = choice["message"]["content"]
            success = True

        except ConnectionError as exc:
            error = str(exc)
            sleep_delay = 10 * (2**exponential_backoffs)
            exponential_backoffs += 1
            await asyncio.sleep(sleep_delay)
        except Exception as exc:
            error = str(exc)
            attempt += 1
            await asyncio.sleep(random.uniform(1.0, 3.0) * attempt)

    latency = time.time() - start

    return {
        "document_id": entry.document_id,
        "page": page_num,
        "page_index": page_index,
        "pdf_path": entry.pdf_path,
        "model": model_name,
        "success": success,
        "latency": latency,
        "usage": usage,
        "content": output_text,
        "error": error,
    }


async def process_document(
    entry: ManifestEntry,
    cfg: BatchConfig,
    model_name: str,
    logger: logging.Logger,
    metrics: Metrics,
    completion_url: str,
    api_key: Optional[str],
    request_semaphore: asyncio.Semaphore,
    prompt_text: str,
) -> List[Dict[str, Any]]:
    local_pdf = ensure_pdf_local(entry.pdf_path, cfg.download_dir, logger)

    pages = entry.pages
    if not pages:
        suffix = local_pdf.suffix.lower()
        pdf_signature = b""
        try:
            with open(local_pdf, "rb") as fh:
                pdf_signature = fh.read(4)
        except Exception:
            pdf_signature = b""

        if suffix == ".pdf" or pdf_signature == b"%PDF":
            if PdfReader is None:
                raise RuntimeError("pypdf is required but not installed")
            with open(local_pdf, "rb") as pdf_stream:
                try:
                    reader = PdfReader(pdf_stream, strict=False)
                except TypeError:
                    # Older PyPDF versions do not support strict kwarg.
                    reader = PdfReader(pdf_stream)  # type: ignore[arg-type]
                try:
                    page_count = len(reader.pages)
                except Exception as exc:
                    logger.warning(
                        "Failed to enumerate pages with pypdf for %s: %s",
                        entry.document_id,
                        exc,
                    )
                    if fitz is None:
                        raise
                    doc = fitz.open(local_pdf)
                    try:
                        page_count = doc.page_count
                    finally:
                        doc.close()
        elif is_png(str(local_pdf)) or is_jpeg(str(local_pdf)):
            page_count = 1
        else:
            page_count = 1
        pages = list(range(1, page_count + 1))

    if cfg.max_pages_per_doc:
        pages = pages[: cfg.max_pages_per_doc]

    page_tasks: Dict[int, asyncio.Task[Dict[str, Any]]] = {}
    async with asyncio.TaskGroup() as tg:
        for index, page_num in enumerate(pages, start=1):
            page_tasks[page_num] = tg.create_task(
                process_page(
                    entry=entry,
                    cfg=cfg,
                    model_name=model_name,
                    logger=logger,
                    page_num=page_num,
                    page_index=index,
                    local_pdf=local_pdf,
                    completion_url=completion_url,
                    api_key=api_key,
                    request_semaphore=request_semaphore,
                    prompt_text=prompt_text,
                )
            )

    results: List[Dict[str, Any]] = []
    for page_num in sorted(page_tasks.keys()):
        task = page_tasks[page_num]
        try:
            page_result = task.result()
        except Exception as exc:
            logger.error(
                "Unhandled error processing %s page %s: %s",
                entry.document_id,
                page_num,
                exc,
            )
            page_result = {
                "document_id": entry.document_id,
                "page": page_num,
                "page_index": page_num,
                "pdf_path": entry.pdf_path,
                "model": model_name,
                "success": False,
                "latency": 0.0,
                "usage": {},
                "content": None,
                "error": f"task_failed: {exc}",
            }

        metrics.record(page_result["success"], page_result["latency"])
        results.append(page_result)

    return results


async def run_batch(
    cfg: BatchConfig,
    model_name: str,
    base_url: str,
    doc_semaphore: asyncio.Semaphore,
    request_semaphore: asyncio.Semaphore,
    logger: logging.Logger,
) -> None:
    metrics = Metrics()

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.errored_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.download_dir.mkdir(parents=True, exist_ok=True)
    intermediate_success = Path("/data/page_results/success")
    intermediate_errors = Path("/data/page_results/errors")
    shutil.rmtree(intermediate_success.parent, ignore_errors=True)
    intermediate_success.mkdir(parents=True, exist_ok=True)
    intermediate_errors.mkdir(parents=True, exist_ok=True)

    # Get prompt text based on mode
    prompt_text = PROMPT_TEMPLATES.get(cfg.prompt_mode, PROMPT_TEMPLATES["layout-all"])
    logger.info(f"Using prompt mode: {cfg.prompt_mode}")

    with cfg.output_path.open("w", encoding="utf-8") as out_fp, cfg.errored_path.open(
        "w", encoding="utf-8"
    ) as err_fp:
        queue: asyncio.Queue[Optional[ManifestEntry]] = asyncio.Queue()
        write_lock = asyncio.Lock()
        completion_url = f"{base_url.rstrip('/')}/v1/chat/completions"
        api_key = os.environ.get("DOTS_API_KEY")

        with cfg.input_manifest.open("r", encoding="utf-8") as manifest_fp:
            matched_filenames: set[str] = set()
            for line in manifest_fp:
                line = line.strip()
                if not line:
                    continue
                entry = ManifestEntry.from_jsonl(line)
                if cfg.target_pdf_filenames is not None:
                    pdf_filename = extract_pdf_filename(entry.pdf_path)
                    if pdf_filename not in cfg.target_pdf_filenames:
                        continue
                    matched_filenames.add(pdf_filename)
                queue.put_nowait(entry)
        if cfg.target_pdf_filenames is not None:
            missing = cfg.target_pdf_filenames - matched_filenames
            if missing:
                logger.warning(
                    "No manifest entries matched the requested PDFs: %s",
                    ", ".join(sorted(missing)),
                )

        async def worker(worker_id: int) -> None:
            while True:
                await doc_semaphore.acquire()
                entry = await queue.get()
                if entry is None:
                    doc_semaphore.release()
                    queue.task_done()
                    break

                try:
                    page_results = await process_document(
                        entry=entry,
                        cfg=cfg,
                        model_name=model_name,
                        logger=logger,
                        metrics=metrics,
                        completion_url=completion_url,
                        api_key=api_key,
                        request_semaphore=request_semaphore,
                        prompt_text=prompt_text,
                    )
                except Exception as exc:
                    logger.error("Worker %s failed processing %s: %s", worker_id, entry.document_id, exc)
                    # Create an error record for the entire document
                    page_results = [
                        {
                            "document_id": entry.document_id,
                            "page": 0,
                            "page_index": 0,
                            "pdf_path": entry.pdf_path,
                            "model": model_name,
                            "success": False,
                            "latency": 0.0,
                            "usage": {},
                            "content": None,
                            "error": f"document_failed: {exc}",
                        }
                    ]
                    # Record this as a failed "page" for metrics
                    metrics.record(False, 0.0)
                finally:
                    doc_semaphore.release()

                try:
                    for page_result in page_results:
                        async with write_lock:
                            target_dir = intermediate_success if page_result["success"] else intermediate_errors
                            doc_dir = target_dir / page_result["document_id"]
                            doc_dir.mkdir(parents=True, exist_ok=True)
                            page_file = doc_dir / f"page_{page_result['page']}.json"
                            with page_file.open("w", encoding="utf-8") as pf:
                                json.dump(page_result, pf, ensure_ascii=False)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker(i)) for i in range(cfg.workers)]

        for _ in range(cfg.workers):
            queue.put_nowait(None)

        await queue.join()
        for task in workers:
            await task

    def _combine(dir_path: Path, target_fp) -> None:
        for doc_dir in sorted(dir_path.iterdir()):
            if not doc_dir.is_dir():
                continue
            for page_file in sorted(doc_dir.glob("page_*.json")):
                with page_file.open("r", encoding="utf-8") as pf:
                    target_fp.write(pf.read().strip())
                    target_fp.write("\n")

    with cfg.output_path.open("w", encoding="utf-8") as out_fp:
        _combine(intermediate_success, out_fp)

    with cfg.errored_path.open("w", encoding="utf-8") as err_fp:
        _combine(intermediate_errors, err_fp)

    shutil.rmtree(intermediate_success.parent, ignore_errors=True)

    logger.info("Batch complete: %s", metrics.as_dict())


def download_model(model_name: str, cache_dir: Path, logger: logging.Logger) -> Path:
    from huggingface_hub import snapshot_download

    model_cache = cache_dir / model_name.replace("/", "_")
    if model_cache.exists() and any(model_cache.iterdir()):
        logger.info("Using cached model at %s", model_cache)
        return model_cache

    logger.info("Downloading model %s to %s", model_name, model_cache)
    snapshot_download(repo_id=model_name, local_dir=str(model_cache), local_dir_use_symlinks=False)
    return model_cache


@app.function(
    image=IMAGE,
    gpu="H100",
    timeout=60 * 60 * 6,
    volumes={"/data": DATA_VOLUME, "/models": MODEL_VOLUME, "/root/.cache/vllm": VLLM_CACHE_VOLUME},
)
async def run_dots_batch(
    manifest_path: str = "/data/manifest.jsonl",
    output_path: str = "/data/dots_predictions.jsonl",
    error_path: str = "/data/dots_errors.jsonl",
    model_name: str = DEFAULT_MODEL,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: Optional[float] = None,
    max_model_len: int = 131072,
    max_pages_per_doc: Optional[int] = None,
    workers: int = 4,
    max_concurrent_docs: Optional[int] = None,
    max_inflight_requests: Optional[int] = None,
    request_timeout: float = 180.0,
    max_retries: int = 3,
    batch_id: Optional[str] = None,
    prompt_mode: str = "layout-all",
    pdf_filenames: Optional[List[str]] = None,
) -> Dict[str, Any]:
    logger = setup_logging(Path("/data/dots_batch.log"))

    cfg = BatchConfig(
        input_manifest=Path(manifest_path),
        output_path=Path(output_path),
        errored_path=Path(error_path),
        max_pages_per_doc=max_pages_per_doc,
        workers=workers,
        max_concurrent_docs=max_concurrent_docs or max(1, min(workers, 8)),
        max_inflight_requests=max_inflight_requests or 128,
        request_timeout=request_timeout,
        max_retries=max_retries,
        batch_id=batch_id,
        prompt_mode=prompt_mode,
        target_pdf_filenames={extract_pdf_filename(name) for name in pdf_filenames}
        if pdf_filenames
        else None,
    )

    logger.info("Starting DoTS.ocr batch with config: %s", cfg)

    model_cache_dir = Path("/models")
    model_path = download_model(model_name, model_cache_dir, logger)

    doc_slots = max(1, min(cfg.max_concurrent_docs, cfg.workers))
    doc_semaphore = asyncio.Semaphore(doc_slots)
    request_semaphore = asyncio.Semaphore(max(1, cfg.max_inflight_requests))
    server_process, monitor_task = launch_vllm_server(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        semaphore=doc_semaphore,
        max_doc_concurrency=doc_slots,
        logger=logger,
    )

    try:
        base_url = f"http://localhost:{VLLM_PORT}"
        await wait_for_vllm(base_url)
        await run_batch(
            cfg=cfg,
            model_name="dots-ocr",
            base_url=base_url,
            doc_semaphore=doc_semaphore,
            request_semaphore=request_semaphore,
            logger=logger,
        )
    finally:
        if server_process.poll() is None:
            logger.info("Terminating vLLM server")
            server_process.terminate()
            try:
                server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("vLLM server did not exit promptly; killing")
                server_process.kill()
        monitor_task.cancel()
        await asyncio.gather(monitor_task, return_exceptions=True)
        shutil.rmtree(cfg.download_dir, ignore_errors=True)

    logger.info("DoTS.ocr batch completed successfully")
    return {"status": "completed", "output_path": str(cfg.output_path), "errors_path": str(cfg.errored_path)}


@app.local_entrypoint()
def main(
    manifest: str = "/data/manifest.jsonl",
    prompt_mode: str = "layout-all",
    workers: int = 4,
    max_concurrent_docs: Optional[int] = None,
    max_inflight_requests: Optional[int] = None,
    pdf_filenames: Optional[str] = None,
) -> None:
    kwargs = {"manifest_path": manifest, "workers": workers, "prompt_mode": prompt_mode}
    if max_concurrent_docs is not None:
        kwargs["max_concurrent_docs"] = max_concurrent_docs
    if max_inflight_requests is not None:
        kwargs["max_inflight_requests"] = max_inflight_requests
    if pdf_filenames:
        tokens = re.split(r"[,\s]+", pdf_filenames.strip())
        selected = [token for token in tokens if token]
        if selected:
            kwargs["pdf_filenames"] = selected

    result = run_dots_batch.remote(**kwargs)
    print(result)
