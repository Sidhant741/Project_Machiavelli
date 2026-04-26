"""
inference.py — Project Machiavelli
=====================================
Run the environment in inference / evaluation mode.

Differences from train.py:
  - No learning updates — pure rollout.
  - Loads a specific difficulty + day range.
  - Produces a detailed evaluation report:
      per-agent accuracy, trust evolution, deception rate,
      survival rate, reward trajectory.
  - Supports a --watch flag to print every agent message in full.
  - Results saved to inference_results/<timestamp>.json

Usage:
    python inference.py --difficulty medium --days 4 --episodes 3 --model llama3.2
    python inference.py --difficulty hard   --days 4 --watch
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import ollama

# Reuse all phase runners and helpers from train.py
try:
    from Train import (
        llm_call,
        agent_system_prompt,
        run_pre_discussion,
        run_task_execution,
        run_post_discussion,
        run_voting,
        compute_rewards,
        share_task_results,
        _parse_trust_delta,
        MODEL, TASK_DIR, LOG_DIR,
    )
    from environment import PMEnvironment
    from models import Phase
    from task_loader import TaskLoader
    from graders.easy import EasyGrader
    from graders.medium import MediumGrader
    from graders.hard import HardGrader
    from utils import evaluate_task_answers, summarise_task_results
    from server.config import GAME_CONFIGS
except ImportError:
    from server.Train import (
        llm_call,
        agent_system_prompt,
        run_pre_discussion,
        run_task_execution,
        run_post_discussion,
        run_voting,
        compute_rewards,
        share_task_results,
        _parse_trust_delta,
        MODEL, TASK_DIR, LOG_DIR,
    )
    from server.environment import PMEnvironment
    from models import Phase
    from server.task_loader import TaskLoader
    from graders.easy import EasyGrader
    from graders.medium import MediumGrader
    from graders.hard import HardGrader
    from server.utils import evaluate_task_answers, summarise_task_results
    from server.config import GAME_CONFIGS


RESULTS_DIR = "inference_results"


# ---------------------------------------------------------------------------
# Evaluation tracker
# ---------------------------------------------------------------------------

class EvalTracker:
    """
    Tracks metrics across all days and episodes for a single inference run.

    Metrics collected:
      - Per-agent task accuracy (correct / total)
      - Per-agent survival across days
      - Deception rate (fraction of pre-discussion messages that were LIE or TWIST)
      - Trust evolution (trust score snapshots per day)
      - Reward trajectory (total reward per agent per day)
      - Trust-decision stats (used_shared_answer rate, random_value distribution)
    """

    def __init__(self, agent_ids: List[int]) -> None:
        self.agent_ids = agent_ids

        # { agent_id: { "correct": int, "total": int } }
        self.task_accuracy: Dict[int, Dict[str, int]] = {
            aid: {"correct": 0, "total": 0} for aid in agent_ids
        }
        # { agent_id: { "survived": int, "days": int } }
        self.survival: Dict[int, Dict[str, int]] = {
            aid: {"survived": 0, "days": 0} for aid in agent_ids
        }
        # { agent_id: { "lie": int, "twist": int, "truth": int } }
        self.deception: Dict[int, Dict[str, int]] = {
            aid: {"lie": 0, "twist": 0, "truth": 0} for aid in agent_ids
        }
        # { agent_id: [ (day, trust_dict) ] }
        self.trust_snapshots: Dict[int, List[Tuple[int, Dict]]] = defaultdict(list)

        # { agent_id: [ (day, reward) ] }
        self.reward_trajectory: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

        # trust-decision stats
        self.used_shared_answer: Dict[int, int] = defaultdict(int)
        self.solved_by_self:     Dict[int, int] = defaultdict(int)
        self.random_values:      Dict[int, List[float]] = defaultdict(list)

    def record_task(self, agent_id: int, is_correct: bool) -> None:
        self.task_accuracy[agent_id]["total"]   += 1
        self.task_accuracy[agent_id]["correct"] += int(is_correct)

    def record_survival(self, agent_id: int, survived: bool) -> None:
        self.survival[agent_id]["days"]     += 1
        self.survival[agent_id]["survived"] += int(survived)

    def record_deception(self, agent_id: int, veracity: str) -> None:
        key = veracity.lower()
        if key in self.deception.get(agent_id, {}):
            self.deception[agent_id][key] += 1

    def record_trust_snapshot(self, agent_id: int, day: int, trust: Dict) -> None:
        self.trust_snapshots[agent_id].append((day, dict(trust)))

    def record_reward(self, agent_id: int, day: int, reward: float) -> None:
        self.reward_trajectory[agent_id].append((day, reward))

    def record_trust_decision(self, agent_id: int, decision: Dict) -> None:
        if decision.get("used_shared_answer"):
            self.used_shared_answer[agent_id] += 1
        else:
            self.solved_by_self[agent_id] += 1
        rv = decision.get("random_value")
        if rv is not None:
            self.random_values[agent_id].append(rv)

    def accuracy(self, agent_id: int) -> float:
        d = self.task_accuracy.get(agent_id, {})
        t = d.get("total", 0)
        return round(d.get("correct", 0) / t, 4) if t else 0.0

    def survival_rate(self, agent_id: int) -> float:
        d = self.survival.get(agent_id, {})
        t = d.get("days", 0)
        return round(d.get("survived", 0) / t, 4) if t else 0.0

    def deception_rate(self, agent_id: int) -> float:
        d = self.deception.get(agent_id, {})
        total = sum(d.values())
        lies  = d.get("lie", 0) + d.get("twist", 0)
        return round(lies / total, 4) if total else 0.0

    def avg_random_value(self, agent_id: int) -> Optional[float]:
        vals = self.random_values.get(agent_id, [])
        return round(sum(vals) / len(vals), 4) if vals else None

    def report(self) -> Dict[str, Any]:
        return {
            aid: {
                "task_accuracy":      self.accuracy(aid),
                "survival_rate":      self.survival_rate(aid),
                "deception_rate":     self.deception_rate(aid),
                "used_shared_answer": self.used_shared_answer.get(aid, 0),
                "solved_by_self":     self.solved_by_self.get(aid, 0),
                "avg_random_value":   self.avg_random_value(aid),
                "reward_trajectory":  self.reward_trajectory.get(aid, []),
                "trust_snapshots":    self.trust_snapshots.get(aid, []),
                "deception_counts":   self.deception.get(aid, {}),
            }
            for aid in self.agent_ids
        }

    def print_report(self) -> None:
        sep = "═" * 66
        print(f"\n{sep}")
        print("  INFERENCE EVALUATION REPORT")
        print(sep)
        for aid in self.agent_ids:
            print(f"\n  Agent {aid}")
            print(f"    Task accuracy      : {self.accuracy(aid):.0%}")
            print(f"    Survival rate      : {self.survival_rate(aid):.0%}")
            print(f"    Deception rate     : {self.deception_rate(aid):.0%}  "
                  f"{self.deception.get(aid, {})}")
            print(f"    Used shared answer : {self.used_shared_answer.get(aid, 0)} times")
            print(f"    Solved by self     : {self.solved_by_self.get(aid, 0)} times")
            print(f"    Avg random value   : {self.avg_random_value(aid)}")
            rewards = [r for _, r in self.reward_trajectory.get(aid, [])]
            total_r = round(sum(rewards), 4)
            print(f"    Total reward       : {total_r:+.3f}  trajectory={rewards}")
        print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Collect pre-discussion deception stats
# ---------------------------------------------------------------------------

def collect_deception_stats(env: PMEnvironment, tracker: EvalTracker, day: int) -> None:
    """Read Phase 2 messages from env state and record veracity counts."""
    day_msgs = env.state.pre_task_messages.get(day, {})
    for aid, msg in day_msgs.items():
        if aid in tracker.agent_ids:
            tracker.record_deception(aid, msg.veracity.value)


# ---------------------------------------------------------------------------
# Collect trust-decision stats from compression store
# ---------------------------------------------------------------------------

def collect_trust_decision_stats(
    env: PMEnvironment,
    tracker: EvalTracker,
    day: int,
) -> None:
    day_store = env.summary_store.get(f"day_{day}", {})
    for aid, summary in day_store.items():
        td = summary.get("task_decision", {})
        if td:
            tracker.record_trust_decision(aid, td)


# ---------------------------------------------------------------------------
# Single episode rollout
# ---------------------------------------------------------------------------

def run_episode(
    env: PMEnvironment,
    loader: TaskLoader,
    difficulty: str,
    n_days: int,
    models: List[str],
    tracker: EvalTracker,
    episode: int,
    watch: bool = False,
) -> Dict[str, Any]:
    """Run one full episode and return the episode log."""

    obs        = env.reset(task=difficulty)
    cfg        = GAME_CONFIGS[difficulty]
    n_options  = cfg["n_options"]
    correct_pts = cfg["correct_answer_points"]
    episode_log: Dict[str, Any] = {
        "episode":    episode,
        "difficulty": difficulty,
        "days":       [],
    }

    # ── Map agents to models ──────────────────────────────
    all_ids = list(env.agents.keys())
    models_dict = {
        aid: models[i % len(models)]
        for i, aid in enumerate(all_ids)
    }

    print(f"  AGENT → MODEL MAPPING:")
    for aid, mname in models_dict.items():
        print(f"    Agent {aid}: {mname}")

    for day in range(1, n_days + 1):
        if env.is_done:
            break

        alive = env.state.alive_agents
        print(f"\n  {'─'*64}")
        print(f"  [Episode {episode}] DAY {day}  |  Alive: {alive}")
        print(f"  {'─'*64}")

        # Record survival start-of-day
        for aid in tracker.agent_ids:
            if aid in alive:
                tracker.survival[aid]["days"] += 1

        # ── Load tasks ───────────────────────────────────────────────
        questions = loader.get_day_questions(difficulty, day, n_agents=len(alive))
        agent_questions: Dict[int, Dict] = {
            aid: q for aid, q in zip(alive, questions)
        }

        for agent_id, q in agent_questions.items():
            priv, correct, opts = loader.build_private_info(q, n_options)
            env.state.each_agent_private_info[agent_id] = priv
            env.agents[agent_id].private_info = priv
            env.ctx.day_questions[agent_id] = (q["question"], correct, opts)

        # ── Phase 2 ──────────────────────────────────────────────────
        run_pre_discussion(env, models_dict, day, difficulty)
        collect_deception_stats(env, tracker, day)

        # ── Phase 3 ──────────────────────────────────────────────────
        raw_answers  = run_task_execution(env, models_dict, agent_questions, day, difficulty)
        eval_results = evaluate_task_answers(raw_answers, agent_questions, correct_pts)
        eval_summary = summarise_task_results(eval_results)
        share_task_results(env, eval_summary)

        # Record task accuracy
        for aid, result in eval_results.items():
            if aid in tracker.agent_ids:
                tracker.record_task(aid, result.is_correct)

        # ── Phase 4 ──────────────────────────────────────────────────
        run_post_discussion(env, models_dict, eval_summary, day, difficulty)

        # Trust snapshot after Phase 4
        for aid in alive:
            if aid in tracker.agent_ids:
                tracker.record_trust_snapshot(
                    aid, day, env.state.trust_scores_dict.get(aid, {})
                )

        # ── Phase 5 ──────────────────────────────────────────────────
        vote_reasons = run_voting(env, models_dict, eval_summary, day, difficulty)

        # ── Rewards ──────────────────────────────────────────────────
        rewards = compute_rewards(env, eval_results, vote_reasons, None, day, difficulty)
        for aid, r in rewards.items():
            if aid in tracker.agent_ids:
                tracker.record_reward(aid, day, r)

        # Trust-decision stats from compression store
        collect_trust_decision_stats(env, tracker, day)

        # Survival record
        survived_this_day = env.state.alive_agents
        eliminated = (
            env.state.vote_history[-1].eliminated_id
            if env.state.vote_history else None
        )
        for aid in tracker.agent_ids:
            if aid in alive:
                tracker.survival[aid]["survived"] += int(aid in survived_this_day)
                # Undo the double-counting increment done at start
                tracker.survival[aid]["survived"] = min(
                    tracker.survival[aid]["survived"],
                    tracker.survival[aid]["days"],
                )

        if eliminated is not None:
            print(f"\n  ❌  Agent {eliminated} eliminated.")

        episode_log["days"].append({
            "day":          day,
            "alive_start":  alive,
            "alive_end":    list(env.state.alive_agents),
            "task_summary": eval_summary,
            "rewards":      rewards,
            "eliminated":   eliminated,
        })

    episode_log["survivors"] = list(env.state.alive_agents)
    return episode_log


# ---------------------------------------------------------------------------
# Save inference results
# ---------------------------------------------------------------------------

def save_results(
    all_episodes: List[Dict],
    eval_report: Dict,
    difficulty: str,
) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"inference_{difficulty}_{ts}.json")
    out  = {
        "difficulty":   difficulty,
        "timestamp":    ts,
        "eval_report":  eval_report,
        "episodes":     all_episodes,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Results saved → {path}")


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(
    difficulty: str  = "easy",
    n_days: Optional[int] = None,
    n_episodes: int  = 1,
    models: List[str] = [MODEL],
    task_dir: str    = TASK_DIR,
    watch: bool      = False,
    reward_weights: Optional[Dict] = None,
) -> None:
    loader  = TaskLoader(task_dir=task_dir)
    env     = PMEnvironment()
    cfg     = GAME_CONFIGS[difficulty]
    n_agents = cfg["n_agents"]

    if n_days is None:
        n_days = n_agents - 1

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║             Project Machiavelli — Inference / Eval               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  difficulty={difficulty}  days={n_days}  episodes={n_episodes}  models={models}\n")

    tracker      = EvalTracker(agent_ids=list(range(n_agents)))
    all_episodes = []

    for ep in range(1, n_episodes + 1):
        print(f"\n{'═'*66}")
        print(f"  EPISODE {ep}/{n_episodes}")
        print(f"{'═'*66}")

        ep_log = run_episode(
            env=env,
            loader=loader,
            difficulty=difficulty,
            n_days=n_days,
            models=models,
            tracker=tracker,
            episode=ep,
            watch=watch,
        )
        all_episodes.append(ep_log)

        print(f"\n  Episode {ep} survivors: {ep_log['survivors']}")

    # ── Final report ─────────────────────────────────────────────────
    tracker.print_report()
    eval_report = tracker.report()
    save_results(all_episodes, eval_report, difficulty)
    print("  Inference complete ✓\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Machiavelli — Inference")
    parser.add_argument("--difficulty", default="easy",  choices=["easy","medium","hard"])
    parser.add_argument("--episodes",   default=1,       type=int)
    parser.add_argument("--models",     default=[MODEL], nargs="+")
    parser.add_argument("--task_dir",   default=TASK_DIR)
    parser.add_argument("--watch",      action="store_true",
                        help="Print full agent messages to stdout")
    args = parser.parse_args()

    run_inference(
        difficulty=args.difficulty,
        n_episodes=args.episodes,
        models=args.models,
        task_dir=args.task_dir,
        watch=args.watch,
    )