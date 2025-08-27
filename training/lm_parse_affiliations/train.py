from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from prompt import SYSTEM_PROMPT
from reward import format_reward, answer_reward

assert load_dotenv(), "Failed to load environment variables from .env file."

def parse_args():
    parser = ArgumentParser(description="Train a GRPO model on the arXiv affiliation dataset.")

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="The model to use for training. Default is Qwen/Qwen3-4B.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save the trained model checkpoints.",
    )

    parser.add_argument(
        "--learning_rate", "-lr",
        type=float,
        default=1e-5,
        help="Learning rate for the optimizer. Default is 1e-5.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # load dataset
    dataset = load_from_disk('data/arxiv_author_affiliations')
    dataset = dataset.map(
        lambda x: {
            "prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": tokenizer.decode(tokenizer(x["pdf_content"], truncation=True, max_length=7_000).input_ids, skip_special_tokens=True),
                },
            ],
            "answer": x['authors'],
        },
        remove_columns=["doi", "title", "authors", "filename", "pdf_content"],
    )
    # dataset.save_to_disk('data/arxiv_author_affiliations_chat_16k')

    # lora
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.01,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # run
    run_name = f'grpo-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-{args.model.split("/")[-1]}'
    output_dir = Path(args.checkpoint_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # trainer
    config = GRPOConfig(
        output_dir = output_dir,

        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        learning_rate = args.learning_rate,
        lr_scheduler_type = "cosine",
        warmup_ratio = 0.03,

        max_prompt_length = 8_000,
        max_completion_length = 16_000,

        scale_rewards = False,
        loss_type = "dr_grpo",

        logging_steps=10,
        save_steps=50,
        log_completions = True,
        run_name = run_name,
        report_to = 'wandb',
        use_vllm = False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, answer_reward],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # start training
    trainer.train("checkpoints/grpo-2025-08-03-11-51-Qwen3-4B/checkpoint-200")
    trainer.save_model(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")

    print(f"Training completed. Model saved to {output_dir / 'final'}.")


if __name__ == "__main__":
    main()
