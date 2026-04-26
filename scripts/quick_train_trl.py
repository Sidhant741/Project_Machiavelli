"""
Fast offline TRL training from collected trajectories.

This is intentionally simple for a 1-hour pipeline:
1) Load JSONL created by `collect_trajectories.py`
2) Keep top-reward samples (rejection-style filtering)
3) Run TRL SFT on prompt -> completion pairs

Usage:
  python scripts/quick_train_trl.py \
    --input logs/trajectories.jsonl \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir outputs/machiavelli-sft \
    --steps 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt", "")).strip()
            completion = str(row.get("completion", "")).strip()
            if not prompt or not completion:
                continue
            if completion.startswith("[LLM ERROR"):
                continue
            rows.append(row)
    return rows


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int((len(xs) - 1) * q)
    return xs[max(0, min(idx, len(xs) - 1))]


def to_dataset(rows: List[Dict], min_reward_quantile: float) -> Dataset:
    rewards = [float(r.get("episode_reward", 0.0)) for r in rows]
    threshold = percentile(rewards, min_reward_quantile)

    kept = []
    for r in rows:
        reward = float(r.get("episode_reward", 0.0))
        if reward < threshold:
            continue
        text = (
            "### Prompt\n"
            f"{r['prompt'].strip()}\n\n"
            "### Response\n"
            f"{r['completion'].strip()}"
        )
        kept.append({"text": text, "reward": reward, "phase": r.get("phase", "unknown")})

    if not kept:
        # Fallback: keep everything instead of failing.
        for r in rows:
            kept.append(
                {
                    "text": (
                        "### Prompt\n"
                        f"{r['prompt'].strip()}\n\n"
                        "### Response\n"
                        f"{r['completion'].strip()}"
                    ),
                    "reward": float(r.get("episode_reward", 0.0)),
                    "phase": r.get("phase", "unknown"),
                }
            )

    return Dataset.from_list(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast TRL SFT from Machiavelli trajectories.")
    parser.add_argument("--input", default="logs/trajectories.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", default="outputs/machiavelli-sft")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--reward_quantile", type=float, default=0.6)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    rows = load_rows(input_path)
    if not rows:
        raise RuntimeError("No usable rows found in input JSONL.")
    ds = to_dataset(rows, min_reward_quantile=args.reward_quantile)

    print(f"Loaded rows: {len(rows)}")
    print(f"Training rows after filtering: {len(ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=100,
        max_steps=args.steps,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        args=training_args,
        max_seq_length=1024,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model to: {args.output_dir}")


if __name__ == "__main__":
    main()
