"""
Quick trajectory collector for Project Machiavelli.

What it does:
- Runs the existing `server/Train.py` training loop.
- Intercepts every LLM call (prompt + completion).
- Joins calls with day-level rewards from `logs/epXXX_<difficulty>.json`.
- Writes a flat JSONL suitable for fast offline training.

Usage:
  python collect_trajectories.py --difficulty easy --episodes 20 --models llama3.2 qwen2.5:0.5b
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = REPO_ROOT / "logs"


def _parse_agent_id(system_prompt: str) -> Optional[int]:
    match = re.search(r"You are Agent\s+(\d+)", system_prompt)
    return int(match.group(1)) if match else None


def _parse_day(system_prompt: str) -> Optional[int]:
    match = re.search(r"Day\s+(\d+)", system_prompt)
    return int(match.group(1)) if match else None


def _parse_phase(user_prompt: str) -> str:
    upper = user_prompt.upper()
    if "PHASE 2" in upper:
        return "phase_2_pre_discussion"
    if "PHASE 3" in upper:
        return "phase_3_task_execution"
    if "PHASE 4" in upper and "TRUST ASSESSMENT" in upper:
        return "phase_4_trust_assessment"
    if "PHASE 4" in upper:
        return "phase_4_post_discussion"
    if "PHASE 5" in upper:
        return "phase_5_voting"
    return "unknown"


def _load_reward_index(difficulty: str, episodes: int) -> Dict[tuple, float]:
    """
    Returns {(episode, day, agent_id): reward}.
    """
    index: Dict[tuple, float] = {}
    for ep in range(1, episodes + 1):
        path = LOG_DIR / f"ep{ep:03d}_{difficulty}.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for day_item in payload.get("days", []):
            day = int(day_item.get("day", 0))
            rewards = day_item.get("rewards", {}) or {}
            for aid_str, reward in rewards.items():
                try:
                    aid = int(aid_str)
                except Exception:
                    aid = int(aid_str) if str(aid_str).isdigit() else None
                if aid is None:
                    continue
                index[(ep, day, aid)] = float(reward)
    return index


def collect(
    difficulty: str,
    episodes: int,
    models: List[str],
    output_file: str,
) -> None:
    # Late import so script can run from repo root cleanly.
    from server import Train as T

    interactions: List[Dict[str, Any]] = []
    orig_llm_call = T.llm_call

    state = {"episode": 1, "last_day": None}

    def wrapped_llm_call(
        model: str,
        system: str,
        user: str,
        max_tokens: int = T.MAX_TOKENS,
        images: Optional[List[str]] = None,
    ) -> str:
        response = orig_llm_call(model, system, user, max_tokens=max_tokens, images=images)
        agent_id = _parse_agent_id(system) or -1
        day = _parse_day(system) or 1
        phase = _parse_phase(user)

        last_day = state["last_day"]
        if last_day is not None and day == 1 and last_day > 1:
            state["episode"] += 1
        state["last_day"] = day

        interactions.append(
            {
                "episode": state["episode"],
                "day": day,
                "agent_id": agent_id,
                "phase": phase,
                "model": model,
                "prompt": user.strip(),
                "completion": (response or "").strip(),
            }
        )
        return response

    T.llm_call = wrapped_llm_call
    try:
        T.train(
            difficulty=difficulty,
            n_episodes=episodes,
            models=models,
            task_dir=T.TASK_DIR,
        )
    finally:
        T.llm_call = orig_llm_call

    reward_index = _load_reward_index(difficulty=difficulty, episodes=episodes)
    for item in interactions:
        key = (item["episode"], item["day"], item["agent_id"])
        item["episode_reward"] = reward_index.get(key, 0.0)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in interactions:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote {len(interactions)} records to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Machiavelli trajectories to JSONL.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--models", nargs="+", default=["llama3.2"])
    parser.add_argument("--output", default="logs/trajectories.jsonl")
    args = parser.parse_args()

    collect(
        difficulty=args.difficulty,
        episodes=args.episodes,
        models=args.models,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
