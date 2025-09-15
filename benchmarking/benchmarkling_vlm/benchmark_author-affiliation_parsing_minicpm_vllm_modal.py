import io
import os
import re
import sys
import json
import math
import time
import base64
import asyncio
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import modal
from tqdm import tqdm


app = modal.App(
    "benchmark-minicpm-vlm-v2",
    secrets=[modal.Secret.from_dotenv()],
)

data_volume = modal.Volume.from_name(
    "affiliation-data-vlm", create_if_missing=True)
model_volume = modal.Volume.from_name("vlm-models", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name(
    "vllm-cache", create_if_missing=True)

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "wget")
    .pip_install(
        "torch==2.7.1",
        "huggingface_hub[hf_transfer]==0.34.4",
        "transformers",
        "pillow",
        "openai>=1.99.1",
        "tqdm",
        "aiohttp",
        "aiofiles",
    )
    .run_commands(
        "git clone https://github.com/vllm-project/vllm.git /tmp/vllm",
        "cd /tmp/vllm && VLLM_USE_PRECOMPILED=1 pip install --editable .",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_CACHE_ROOT": "/root/.cache/vllm",
        "HF_TOKEN": os.environ.get("HF_TOKEN"),
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",  # Disable v1 engine for vision models
    })
)


def setup_logging(log_file: str = "/data/vllm_errors.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(
            log_file), logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_prompt_template(prompt_file: Path) -> str:
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        raise RuntimeError(f"Failed to load prompt template: {e}")


def parse_json_output(output: str) -> Optional[List[Dict]]:
    try:
        match = re.search(r"```json\s*([\s\S]*?)\s*```", output)
        if match:
            return json.loads(match.group(1))
        start_idx = output.find('[')
        end_idx = output.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return json.loads(output[start_idx:end_idx + 1])
        return json.loads(output)
    except (json.JSONDecodeError, AttributeError):
        return None


def save_result_incrementally(result: Dict, output_file: str):
    with open(output_file, 'a', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')


@app.function(
    image=vllm_image,
    gpu="H100",
    timeout=60 * 60 * 4,
    volumes={
        "/data": data_volume,
        "/models": model_volume,
        "/root/.cache/vllm": vllm_cache_volume
    },
    retries=1,
    max_containers=1,
)
async def run_inference(
    batch_size: int = 1,
    num_samples: Optional[int] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    max_side: int = 2000,
    max_megapixels: float = 3.5,
    grayscale: bool = False,
    image_format: str = "JPEG",
    jpeg_quality: int = 92,
):
    import aiohttp
    import aiofiles
    from huggingface_hub import snapshot_download
    from openai import AsyncOpenAI
    from PIL import Image

    logger = setup_logging()

    model_id = "openbmb/MiniCPM-V-4_5"
    model_cache_path = Path("/models") / model_id.replace("/", "_")

    if not model_cache_path.exists():
        logger.info(f"Downloading model {model_id} to cache...")
        model_cache_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=model_id, local_dir=str(
            model_cache_path), local_dir_use_symlinks=False, token=os.environ.get("HF_TOKEN"))

    cmd = [
        "vllm", "serve", str(
            model_cache_path), "--host", "0.0.0.0", "--port", "8000",
        "--served-model-name", model_id, "--tensor-parallel-size", "1", "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.95", "--trust-remote-code", "--disable-frontend-multiprocessing",
    ]
    logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")
    server_process = subprocess.Popen(" ".join(cmd), shell=True)

    server_url = "http://localhost:8000"
    for i in range(120):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{server_url}/health") as resp:
                    if resp.status == 200:
                        logger.info("vLLM server is ready!")
                        break
        except aiohttp.ClientConnectorError:
            await asyncio.sleep(2)
            if i == 119:
                server_process.kill()
                raise RuntimeError("vLLM server failed to start in time.")

    input_file, output_file, prompt_file = "/data/metadata.json", "/data/predictions.jsonl", "/data/prompt.txt"
    image_base_dir = "/data/training_pdf_images"

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input metadata file not found: {input_file}")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if num_samples:
        data = data[:num_samples]
    system_prompt = load_prompt_template(Path(prompt_file))

    client = AsyncOpenAI(base_url=f"{server_url}/v1", api_key="dummy", max_retries=3)
    if os.path.exists(output_file):
        os.remove(output_file)

    def _compute_scale(w: int, h: int) -> float:
        side_scale = 1.0 if max(w, h) <= max_side else (
            max_side / float(max(w, h)))
        mp_limit = max_megapixels * 1_000_000.0
        mp_scale = 1.0 if (
            w * h) <= mp_limit else math.sqrt(mp_limit / float(w * h))
        return min(side_scale, mp_scale)

    async def process_image_to_bytes(img_path: Path) -> bytes:
        async with aiofiles.open(img_path, "rb") as f:
            img_bytes = await f.read()
        img = Image.open(io.BytesIO(img_bytes))

        scale = _compute_scale(img.width, img.height)
        if scale < 1.0:
            new_w, new_h = max(1, int(img.width * scale)
                               ), max(1, int(img.height * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
        if grayscale:
            img = img.convert("L")

        buf = io.BytesIO()
        fmt = image_format.upper()
        if fmt == "JPEG":
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        else:
            img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    async def process_entry(entry: Dict) -> Dict:
        arxiv_id, filename_base = entry['arxiv_id'], Path(
            entry['filename']).stem
        image_dir = Path(image_base_dir) / filename_base
        start_time = time.time()

        if not image_dir.is_dir():
            return {'arxiv_id': arxiv_id, 'predicted_authors': None, 'error': 'Image directory not found'}
        image_paths = sorted(list(image_dir.glob(f"{filename_base}_page_*.png")))
        if not image_paths:
            return {'arxiv_id': arxiv_id, 'predicted_authors': None, 'error': 'No images found in directory'}

        try:
            content = [{"type": "text", "text": system_prompt}]
            for img_path in image_paths:
                # **MODIFICATION**
                processed_img_bytes = await process_image_to_bytes(img_path)
                b64_img = base64.b64encode(processed_img_bytes).decode("utf-8")
                mime = "image/jpeg" if image_format.upper() == "JPEG" else "image/png"
                content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_img}"}})

            response = await client.chat.completions.create(
                model=model_id, messages=[
                    {"role": "user", "content": content}],
                max_tokens=max_tokens, temperature=temperature,
            )
            output_text = response.choices[0].message.content
            parsed_output = parse_json_output(output_text)

            return {'arxiv_id': arxiv_id, 'predicted_authors': parsed_output, 'raw_output': output_text,
                    'error': None if parsed_output else 'Failed to parse JSON', 'processing_time': time.time() - start_time}
        except Exception as e:
            logger.error(f"Error processing {arxiv_id}: {str(e)}")
            return {'arxiv_id': arxiv_id, 'predicted_authors': None, 'error': str(e), 'processing_time': time.time() - start_time}

    all_results = []
    pbar = tqdm(total=len(data), desc="Benchmarking Documents")
    for i in range(0, len(data), batch_size):
        batch_results = await asyncio.gather(*(process_entry(entry) for entry in data[i:i + batch_size]))
        for result in batch_results:
            all_results.append(result)
            save_result_incrementally(result, output_file)
        pbar.update(len(batch_results))
    pbar.close()
    server_process.terminate()

    total = len(all_results)
    successful = sum(1 for r in all_results if r.get(
        'predicted_authors') is not None)
    logger.info(f"\nInference Complete. Total: {total}, Successful: {successful}")
    return {"total": total, "successful": successful}


@app.function(image=vllm_image, volumes={"/data": data_volume}, timeout=60 * 30)
def _write_file_to_volume(remote_path: str, content: bytes):
    full_path = Path("/data") / remote_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "wb") as f:
        f.write(content)


@app.function(image=vllm_image, volumes={"/data": data_volume}, timeout=600)
def download_results():
    results_path = "/data/predictions.jsonl"
    if not os.path.exists(results_path):
        return None, "Results file not found."
    with open(results_path, 'rb') as f:
        content = f.read()
    summary = f"Downloaded {len(content.splitlines())} results."
    return content, summary


@app.function(image=vllm_image, volumes={"/data": data_volume}, timeout=600)
def clear_volume(path: str = "."):
    import shutil
    target_path = Path("/data") / path
    if target_path.exists():
        if target_path.is_dir():
            shutil.rmtree(target_path)
            target_path.mkdir()
            return f"Cleared contents of directory: {target_path}"
        elif target_path.is_file():
            target_path.unlink()
            return f"Deleted file: {target_path}"
    return "Path not found."


@app.function(image=vllm_image, volumes={"/models": model_volume}, timeout=600)
def clear_model_cache(model_name: Optional[str] = None):
    import shutil
    if model_name:
        model_path = Path("/models") / model_name.replace("/", "_")
        cleared = []
        if model_path.exists():
            shutil.rmtree(model_path)
            cleared.append(f"Model: {model_path}")
        return f"Cleared: {', '.join(cleared)}" if cleared else f"No cache found for {model_name}"
    else:
        models_dir = Path("/models")
        if models_dir.exists():
            shutil.rmtree(models_dir)
            models_dir.mkdir(parents=True, exist_ok=True)
            return "Cleared all model caches."
        return "No model caches found."


@app.local_entrypoint()
def main(
    upload: bool = False, image_dir: str = "", input_file: str = "", prompt_file: str = "",
    download: bool = False, clear: bool = False, clear_cache: bool = False, model_name: str = "openbmb/MiniCPM-V-4_5",
    batch_size: int = 1, num_samples: Optional[int] = None
):
    if upload:
        if not all([image_dir, input_file, prompt_file]):
            print(
                "Error: For upload, provide --image-dir, --input-file, and --prompt-file.")
            return
        for local, remote in [(input_file, "metadata.json"), (prompt_file, "prompt.txt")]:
            print(f"Uploading {local} to /data/{remote}...")
            with open(local, "rb") as f:
                _write_file_to_volume.remote(remote, f.read())

        print(f"Uploading image directory {image_dir}...")
        image_dir_path = Path(image_dir)
        for local_path in tqdm(list(image_dir_path.glob("**/*.png")), desc="Uploading images"):
            remote_path = local_path.relative_to(image_dir_path.parent)
            with open(local_path, "rb") as f:
                _write_file_to_volume.remote(str(remote_path), f.read())
        print("Data upload complete.")
    elif download:
        print("Downloading results...")
        content, summary = download_results.remote()
        if content is None:
            print(summary)
            return
        output_file = f"minicpm_predictions_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        with open(output_file, 'wb') as f:
            f.write(content)
        print(summary)
        print(f"Results saved to {output_file}")
    elif clear:
        print("Clearing all data from the volume...")
        print(clear_volume.remote("."))
    elif clear_cache:
        print(f"Clearing model cache for '{model_name}'...")
        print(clear_model_cache.remote(model_name))
    else:
        print("Starting vLLM benchmark on Modal...")
        result = run_inference.remote(
            batch_size=batch_size, num_samples=num_samples)
        print("\n--- Benchmark Summary ---")
        print(f"Total entries processed: {result['total']}")
        print(f"Successful extractions: {result['successful']}")
