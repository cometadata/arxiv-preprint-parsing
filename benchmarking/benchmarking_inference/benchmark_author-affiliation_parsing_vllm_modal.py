import os
import sys
import json
import time
import modal
import logging
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

app = modal.App("benchmark-affiliation-vllm")

data_volume = modal.Volume.from_name("affiliation-data", create_if_missing=True)
model_volume = modal.Volume.from_name("affiliation-models", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("vllm-cache", create_if_missing=True)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "wget")
    .pip_install(
        "torch==2.7.1",
        "vllm==0.10.1.1",
        "huggingface_hub[hf_transfer]==0.34.4",
        "openai>=1.99.1",
        "tqdm",
        "aiofiles",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_USE_V1": "1",
        "VLLM_CACHE_ROOT": "/root/.cache/vllm",
    })
)


def setup_logging(log_file: str = "/data/vllm_errors.log"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
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
        start_idx = output.find('[')
        end_idx = output.rfind(']')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = output[start_idx:end_idx + 1]
            return json.loads(json_str)
        else:
            return json.loads(output)
    except json.JSONDecodeError:
        return None
    except Exception:
        return None


def save_result_incrementally(result: Dict, output_file: str):
    with open(output_file, 'a', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')


BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B")
LORA_PATH = os.environ.get("LORA_PATH", "cometadata/affiliation-parsing-lora-Qwen3-4B")
MAX_LORA_RANK = int(os.environ.get("MAX_LORA_RANK", "64"))
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
FAST_BOOT = os.environ.get("FAST_BOOT", "true").lower() == "true"
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "false").lower() == "true"
LOAD_IN_8BIT = os.environ.get("LOAD_IN_8BIT", "false").lower() == "true"


@app.function(
    image=vllm_image,
    gpu="H100",
    memory=32768,
    timeout=60 * 60 * 3,
    volumes={
        "/data": data_volume,
        "/models": model_volume,
        "/root/.cache/vllm": vllm_cache_volume
    },
    retries=1
)
async def run_inference_with_server(
    batch_size: int = 1,
    max_markdown_length: int = 6000,
    max_tokens: int = 2000,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    do_sample: bool = True,
    num_samples: Optional[int] = None,
    use_lora: bool = True,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    fast_boot: bool = True,
):
    import subprocess
    from huggingface_hub import snapshot_download
    import shutil
    from openai import AsyncOpenAI
    from tqdm import tqdm
    import aiohttp
    
    logger = setup_logging()
    
    base_model = BASE_MODEL
    lora_path = LORA_PATH
    max_lora_rank = MAX_LORA_RANK
    tensor_parallel_size = TENSOR_PARALLEL_SIZE

    base_model_cache = Path("/models") / "base_model" / base_model.replace("/", "_")
    lora_cache = Path("/models") / "lora" / lora_path.replace("/", "_")

    if not base_model_cache.exists():
        logger.info(f"Downloading base model {base_model} to cache...")
        base_model_cache.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=base_model,
            local_dir=str(base_model_cache),
            local_dir_use_symlinks=False
        )
    
    if not lora_cache.exists():
        logger.info(f"Downloading LoRA adapter {lora_path} to cache...")
        lora_cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=lora_path,
                local_dir=str(lora_cache),
                local_dir_use_symlinks=False
            )
        except Exception as e:
            logger.warning(f"Could not download LoRA from HuggingFace: {e}")
            logger.info("LoRA adapter needs to be uploaded to the volume")

    cmd = [
        "vllm", "serve",
        str(base_model_cache),
        "--host", "0.0.0.0",
        "--port", "8000",
        "--served-model-name", base_model,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", "8192",
        "--max-num-seqs", "32",
        "--uvicorn-log-level", "info",
    ]

    if lora_cache.exists():
        cmd.extend([
            "--enable-lora",
            "--lora-modules", f"affiliation-lora={str(lora_cache)}",
            "--max-lora-rank", str(max_lora_rank),
            "--max-cpu-loras", "4",
        ])

    if load_in_4bit or load_in_8bit:
        if load_in_8bit:
            cmd.extend(["--quantization", "fp8"])

    if fast_boot:
        cmd.append("--enforce-eager")
    else:
        cmd.append("--no-enforce-eager")
    
    logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")

    logger.info("Starting vLLM server...")
    server_process = subprocess.Popen(" ".join(cmd), shell=True)

    server_url = "http://localhost:8000"
    max_retries = 60
    for i in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{server_url}/health") as resp:
                    if resp.status == 200:
                        logger.info("vLLM server is ready!")
                        break
        except:
            await asyncio.sleep(2)
            if i == max_retries - 1:
                raise RuntimeError("vLLM server failed to start")

    logger.info("Starting inference...")

    input_file = "/data/merged_output.json"
    output_file = "/data/predictions.jsonl"
    prompt_file = "/data/prompt.txt"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    logger.info(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if num_samples:
        data = data[:num_samples]
    
    logger.info(f"Processing {len(data)} entries...")

    system_prompt = load_prompt_template(Path(prompt_file))

    client = AsyncOpenAI(
        base_url=f"{server_url}/v1",
        api_key="dummy-key",
    )

    if os.path.exists(output_file):
        logger.info(f"Clearing existing output file: {output_file}")
        os.remove(output_file)
    
    all_results = []

    model_name = "affiliation-lora" if use_lora and lora_cache.exists() else base_model

    async def process_entry(entry):
        arxiv_id = entry.get('arxiv_id', 'unknown')
        markdown_text = entry.get('markdown_text', '')
        
        if not markdown_text:
            logger.warning(f"Empty markdown text for {arxiv_id}")
            return {
                'arxiv_id': arxiv_id,
                'predicted_authors': None,
                'raw_output': '',
                'error': 'Empty markdown text',
                'processing_time': 0.0
            }
        
        try:
            if len(markdown_text) > max_markdown_length:
                markdown_text = markdown_text[:max_markdown_length]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": markdown_text}
            ]
            
            start_time = time.time()

            if do_sample:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    extra_body={
                        "top_k": top_k,
                    }
                )
            else:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0,
                )
            
            output_text = response.choices[0].message.content
            processing_time = time.time() - start_time

            parsed_output = parse_json_output(output_text)
            
            if parsed_output is None:
                logger.warning(f"Failed to parse JSON for {arxiv_id}: {output_text[:200]}...")
            
            return {
                'arxiv_id': arxiv_id,
                'predicted_authors': parsed_output,
                'raw_output': output_text,
                'error': None if parsed_output else 'Failed to parse JSON',
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing {arxiv_id}: {str(e)}")
            return {
                'arxiv_id': arxiv_id,
                'predicted_authors': None,
                'raw_output': '',
                'error': str(e),
                'processing_time': 0.0
            }
    
    pbar = tqdm(total=len(data), desc="Processing entries")
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]

        tasks = [process_entry(entry) for entry in batch]

        batch_results = await asyncio.gather(*tasks)

        for result in batch_results:
            all_results.append(result)
            save_result_incrementally(result, output_file)
        
        pbar.update(len(batch))
    
    pbar.close()

    total = len(all_results)
    successful = sum(1 for r in all_results if r['predicted_authors'] is not None)
    failed = total - successful
    avg_time = sum(r['processing_time'] for r in all_results) / total if total > 0 else 0
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Inference Complete!")
    logger.info(f"Total entries: {total}")
    logger.info(f"Successful: {successful} ({successful/total*100:.1f}%)")
    logger.info(f"Failed: {failed} ({failed/total*100:.1f}%)")
    logger.info(f"Average processing time: {avg_time:.2f}s")
    logger.info(f"Results saved to: {output_file}")

    server_process.terminate()
    
    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "avg_processing_time": avg_time
    }


@app.function(
    image=vllm_image,
    timeout=60 * 60 * 3,
    volumes={
        "/data": data_volume,
        "/models": model_volume,
        "/root/.cache/vllm": vllm_cache_volume
    },
    memory=16384,
    retries=1
)
async def run_vllm_inference(
    server_url: str,
    batch_size: int = 1,
    max_markdown_length: int = 6000,
    max_tokens: int = 2000,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    do_sample: bool = False,
    num_samples: Optional[int] = None,
    use_lora: bool = True,
):

    from openai import AsyncOpenAI
    from tqdm import tqdm
    
    logger = setup_logging()

    input_file = "/data/merged_output.json"
    output_file = "/data/predictions.jsonl"
    prompt_file = "/data/prompt.txt"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    logger.info(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if num_samples:
        data = data[:num_samples]
    
    logger.info(f"Processing {len(data)} entries...")

    system_prompt = load_prompt_template(Path(prompt_file))

    client = AsyncOpenAI(
        base_url=f"{server_url}/v1",
        api_key="dummy-key",
    )
    
    if os.path.exists(output_file):
        logger.info(f"Clearing existing output file: {output_file}")
        os.remove(output_file)
    
    all_results = []

    model_name = "affiliation-lora" if use_lora else "Qwen/Qwen3-4B"

    pbar = tqdm(total=len(data), desc="Processing entries")
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        
        for entry in batch:
            arxiv_id = entry.get('arxiv_id', 'unknown')
            markdown_text = entry.get('markdown_text', '')
            
            if not markdown_text:
                logger.warning(f"Empty markdown text for {arxiv_id}")
                result = {
                    'arxiv_id': arxiv_id,
                    'predicted_authors': None,
                    'raw_output': '',
                    'error': 'Empty markdown text',
                    'processing_time': 0.0
                }
                all_results.append(result)
                save_result_incrementally(result, output_file)
                continue
            
            try:
                if len(markdown_text) > max_markdown_length:
                    markdown_text = markdown_text[:max_markdown_length]

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": markdown_text}
                ]
                
                start_time = time.time()

                if do_sample:
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        extra_body={
                            "top_k": top_k,
                        }
                    )
                else:
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0,
                    )
                
                output_text = response.choices[0].message.content
                processing_time = time.time() - start_time

                parsed_output = parse_json_output(output_text)
                
                result = {
                    'arxiv_id': arxiv_id,
                    'predicted_authors': parsed_output,
                    'raw_output': output_text,
                    'error': None if parsed_output else 'Failed to parse JSON',
                    'processing_time': processing_time
                }
                all_results.append(result)
                save_result_incrementally(result, output_file)
                
                if parsed_output is None:
                    logger.warning(f"Failed to parse JSON for {arxiv_id}: {output_text[:200]}...")
                
            except Exception as e:
                logger.error(f"Error processing {arxiv_id}: {str(e)}")
                result = {
                    'arxiv_id': arxiv_id,
                    'predicted_authors': None,
                    'raw_output': '',
                    'error': str(e),
                    'processing_time': 0.0
                }
                all_results.append(result)
                save_result_incrementally(result, output_file)
        
        pbar.update(len(batch))
    
    pbar.close()

    total = len(all_results)
    successful = sum(1 for r in all_results if r['predicted_authors'] is not None)
    failed = total - successful
    avg_time = sum(r['processing_time'] for r in all_results) / total if total > 0 else 0
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Inference Complete!")
    logger.info(f"Total entries: {total}")
    logger.info(f"Successful: {successful} ({successful/total*100:.1f}%)")
    logger.info(f"Failed: {failed} ({failed/total*100:.1f}%)")
    logger.info(f"Average processing time: {avg_time:.2f}s")
    logger.info(f"Results saved to: {output_file}")
    
    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "avg_processing_time": avg_time
    }


@app.function(
    image=vllm_image,
    volumes={"/data": data_volume},
    timeout=600
)
def upload_data_file(file_path: str, file_content: bytes):
    target_path = f"/data/{os.path.basename(file_path)}"
    
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    with open(target_path, 'wb') as f:
        f.write(file_content)
    
    return f"File uploaded successfully to {target_path}"


@app.function(
    image=vllm_image,
    volumes={"/data": data_volume},
    timeout=600
)
def download_results():
    results_path = "/data/predictions.jsonl"
    
    if not os.path.exists(results_path):
        return None, "Results file not found. Please run inference first."
    
    with open(results_path, 'rb') as f:
        content = f.read()
    
    results = []
    with open(results_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    
    total = len(results)
    successful = sum(1 for r in results if r.get('predicted_authors') is not None)
    failed = total - successful
    
    summary = {
        "total": total,
        "successful": successful,
        "failed": failed,
        "success_rate": successful/total*100 if total > 0 else 0
    }
    
    return content, summary


@app.function(
    image=vllm_image,
    volumes={"/models": model_volume},
    timeout=60 * 30
)
def upload_lora_adapter(adapter_path: str, files: Dict[str, bytes]):
    target_dir = f"/models/lora/{os.path.basename(adapter_path)}"
    os.makedirs(target_dir, exist_ok=True)
    
    for filename, content in files.items():
        file_path = os.path.join(target_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(content)
    
    return f"LoRA adapter uploaded successfully to {target_dir}"


@app.function(
    image=vllm_image,
    volumes={"/models": model_volume},
    timeout=600
)
def clear_model_cache(model_name: Optional[str] = None):
    import shutil
    
    if model_name:
        base_path = Path("/models/base_model") / model_name.replace("/", "_")
        lora_path = Path("/models/lora") / model_name.replace("/", "_")
        
        cleared = []
        if base_path.exists():
            shutil.rmtree(base_path)
            cleared.append(f"Base model: {base_path}")
        if lora_path.exists():
            shutil.rmtree(lora_path)
            cleared.append(f"LoRA adapter: {lora_path}")
        
        if cleared:
            return f"Cleared: {', '.join(cleared)}"
        else:
            return f"No cached files found for {model_name}"
    else:
        base_models_dir = Path("/models/base_model")
        lora_dir = Path("/models/lora")
        
        cleared = []
        if base_models_dir.exists():
            shutil.rmtree(base_models_dir)
            base_models_dir.mkdir(parents=True, exist_ok=True)
            cleared.append("All base models")
        if lora_dir.exists():
            shutil.rmtree(lora_dir)
            lora_dir.mkdir(parents=True, exist_ok=True)
            cleared.append("All LoRA adapters")
        
        if cleared:
            return f"Cleared: {', '.join(cleared)}"
        else:
            return "No cached models found"


@app.local_entrypoint()
async def main(
    batch_size: int = 1,
    num_samples: Optional[int] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    do_sample: bool = False,
    use_lora: bool = True,
    fast_boot: bool = True,
    upload: bool = False,
    download: bool = False,
    serve: bool = False,
    clear_cache: bool = False,
    input_file: Optional[str] = None,
    prompt_file: Optional[str] = None,
    lora_dir: Optional[str] = None,
    model_name: Optional[str] = None,
):
    if upload:
        if not input_file or not prompt_file:
            print("Please provide both --input-file and --prompt-file for upload")
            return
        
        print(f"Uploading {input_file}...")
        with open(input_file, 'rb') as f:
            result = upload_data_file.remote(input_file, f.read())
        print(result)
        
        print(f"Uploading {prompt_file}...")
        with open(prompt_file, 'rb') as f:
            result = upload_data_file.remote(prompt_file, f.read())
        print(result)
    
    elif lora_dir:
        print(f"Uploading LoRA adapter from {lora_dir}...")
        files = {}
        for root, _, filenames in os.walk(lora_dir):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, lora_dir)
                with open(file_path, 'rb') as f:
                    files[rel_path] = f.read()
        
        result = upload_lora_adapter.remote(lora_dir, files)
        print(result)
    
    elif serve:
        print("To deploy a standalone vLLM server, use:")
        print("  modal deploy benchmark_vllm_modal.py")
        print("\nThe server will be available at a persistent URL for external access.")
        print("\nFor internal inference (server + client in same container), just run:")
        print("  modal run benchmark_vllm_modal.py --batch-size 4")
    
    elif download:
        print("Downloading results from Modal...")
        content, summary = download_results.remote()
        
        if content is None:
            print(summary)
            return
        
        output_file = "vllm_predictions_from_modal.jsonl"
        with open(output_file, 'wb') as f:
            f.write(content)
        
        print(f"Results saved to {output_file}")
        print("\nSummary:")
        print(f"Total entries: {summary['total']}")
        print(f"Successful: {summary['successful']} ({summary['success_rate']:.1f}%)")
        print(f"Failed: {summary['failed']}")
    
    elif clear_cache:
        print("Clearing model cache on Modal...")
        result = clear_model_cache.remote(model_name)
        print(result)
    
    else:
        print("Starting vLLM inference on Modal...")

        result = run_inference_with_server.remote(
            batch_size=batch_size,
            num_samples=num_samples,
            do_sample=do_sample,
            use_lora=use_lora,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            fast_boot=fast_boot,
        )
        
        print("\nInference Summary:")
        print(f"Total entries: {result['total']}")
        print(f"Successful: {result['successful']} ({result['successful']/result['total']*100:.1f}%)")
        print(f"Failed: {result['failed']}")
        print(f"Average processing time: {result['avg_processing_time']:.2f}s")


