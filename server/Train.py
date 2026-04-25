"""
train.py — Project Machiavelli
=================================
Full training loop. Each agent is a separate LLM instance (one call per agent
per phase action). Uses Ollama as the LLM backend.

Flow per day:
  Phase 1 — Task Reveal    : env loads MCQ from task file, deals private info
  Phase 2 — Pre-Discussion : each agent LLM decides what to share (truth/twist/lie)
  Phase 3 — Task Execution : trust-based decision → agent submits answer
  Phase 4 — Post-Discussion: agents communicate, submit trust assessments
  Phase 5 — Voting         : each agent votes + gives reason
  End-of-day               : grader computes rewards, compression stores summary

Setup:
    pip install ollama
    ollama pull llama3.2

Usage:
    python train.py --difficulty easy --days 4 --model llama3.2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ollama

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from environment import PMEnvironment
    from models import (
        PMAction, ActionType, Phase,
        PreTaskMessage, PostDiscussionMessage,
        TrustAssessment, TrustDelta, MessageVeracity,
    )
    from task_loader import TaskLoader
    from graders.easy import EasyGrader
    from graders.medium import MediumGrader
    from graders.hard import HardGrader
    from utils import evaluate_task_answers, summarise_task_results
    from server.config import GAME_CONFIGS
except ImportError:
    from server.environment import PMEnvironment
    from models import (
        PMAction, ActionType, Phase,
        PreTaskMessage, PostDiscussionMessage,
        TrustAssessment, TrustDelta, MessageVeracity,
    )
    from server.task_loader import TaskLoader
    from graders.easy import EasyGrader
    from graders.medium import MediumGrader
    from graders.hard import HardGrader
    from server.utils import evaluate_task_answers, summarise_task_results
    from server.config import GAME_CONFIGS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL        = "llama3.2"       # Ollama model name
TASK_DIR     = "./tasks"        # directory containing easy/medium/hard .json files
MAX_TOKENS   = 300              # max tokens per LLM response
POST_MSG_CAP = 5                # max messages per pair in Phase 4
LOG_DIR      = "logs"           # where to write episode JSON logs

SEP = "─" * 66


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def llm_call(
    model: str,
    system: str,
    user: str,
    max_tokens: int = MAX_TOKENS,
    images: Optional[List[str]] = None,
) -> str:
    """Call Ollama and return the assistant's response text."""
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user, "images": images or []},
            ],
            options={"num_predict": max_tokens},
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        return f"[LLM ERROR: {e}]"


# ---------------------------------------------------------------------------
# Agent system prompt builder
# ---------------------------------------------------------------------------

def agent_system_prompt(
    agent_id: int,
    alive_agents: List[int],
    trust_scores: Dict[int, float],
    private_info: str,
    day: int,
    difficulty: str,
    history_summary: str = "",
) -> str:
    trust_lines = "\n".join(
        f"  Agent {other}: {trust_scores.get(other, 0.5):.2f}"
        for other in alive_agents if other != agent_id
    )
    mem = f"\nYOUR MEMORY FROM PREVIOUS DAYS:\n{history_summary}" if history_summary else ""
    return (
        f"You are Agent {agent_id} in Project Machiavelli — a strategic social game.\n"
        f"Day {day} | Difficulty: {difficulty} | Alive: {alive_agents}\n\n"
        f"YOUR PRIVATE INFO:\n{private_info}\n\n"
        f"YOUR TRUST SCORES FOR OTHER AGENTS (0=distrust, 1=fully trust):\n{trust_lines}\n\n"
        f"GOAL: Survive. Win points. Avoid elimination.\n"
        f"You may share true or false info. Be strategic.{mem}"
    )


# ---------------------------------------------------------------------------
# Phase 2 — Pre-Discussion LLM action
# ---------------------------------------------------------------------------

def run_pre_discussion(
    env: PMEnvironment,
    models_dict: Dict[int, str],
    day: int,
    difficulty: str,
) -> None:
    """Each alive agent sends one pre-task message (may lie)."""
    alive = env.state.alive_agents

    for agent_id in alive:
        agent  = env.agents[agent_id]
        obs    = env.get_observation(agent_id)
        system = agent_system_prompt(
            agent_id, alive, obs.trust_scores,
            obs.own_private_info, day, difficulty,
            agent.history_summary,
        )

        user = (
            "PHASE 2 — PRE-DISCUSSION\n"
            "Compose ONE message to share with other agents.\n"
            "You MAY share your private info truthfully, twist it slightly, or lie.\n"
            "You must NOT reveal other agents' private info.\n"
            "Then state your veracity label on the last line as:\n"
            "VERACITY: truth | twist | lie\n\n"
            "Write your message now (max 150 words):"
        )

        model_name = models_dict.get(agent_id, "llama3.2")
        raw       = llm_call(model_name, system, user, max_tokens=500)
        content, veracity = _parse_pre_discussion(raw)

        # Pick a random recipient or broadcast
        others    = [a for a in alive if a != agent_id]
        recipient = random.choice([None, random.choice(others)]) if others else None

        msg = PreTaskMessage(
            sender_id=agent_id,
            recipient_id=recipient,
            content=content,
            veracity=veracity,
            day=day,
            private_info_referenced=obs.own_private_info[:100],
        )

        action = PMAction(
            agent_id=agent_id,
            action_type=ActionType.SEND_PRE_TASK_MESSAGE,
            pre_task_message=msg,
        )
        env.step(action)
        print(f"  Agent {agent_id} pre-msg [{veracity.value}]: {content}")


def _parse_pre_discussion(raw: str) -> Tuple[str, MessageVeracity]:
    """Parse LLM output into (content, MessageVeracity)."""
    lines    = raw.strip().splitlines()
    veracity = MessageVeracity.TRUTH

    content_lines = []
    for line in lines:
        if line.strip().upper().startswith("VERACITY:"):
            label = line.split(":", 1)[-1].strip().lower()
            if "lie" in label:
                veracity = MessageVeracity.LIE
            elif "twist" in label:
                veracity = MessageVeracity.TWIST
            else:
                veracity = MessageVeracity.TRUTH
        else:
            content_lines.append(line)

    content = " ".join(content_lines).strip() or raw[:200]
    return content, veracity


# ---------------------------------------------------------------------------
# Phase 3 — Task Execution (trust-based, handled inside env)
# ---------------------------------------------------------------------------

def run_task_execution(
    env: PMEnvironment,
    day: int,
) -> None:
    """
    Submit placeholder answers to trigger Phase 3 env logic.
    The env's trust-based logic (_trust_based_task_decision) will evaluate all m questions 
    per agent and overwrite the input here natively.
    """
    alive   = env.state.alive_agents
    for agent_id in alive:
        action = PMAction(
            agent_id=agent_id,
            action_type=ActionType.SUBMIT_TASK_INPUT,
            task_input="placeholder_to_trigger_phase_advance",
        )
        env.step(action)


# ---------------------------------------------------------------------------
# Phase 3 result sharing — show agents the task results
# ---------------------------------------------------------------------------

def summarise_task_results() -> None: pass

def share_task_results(
    env: PMEnvironment,
) -> None:
    """Print multi-question task results summary to console."""
    print(f"\n  {'─'*50}")
    task_res = env.state.task_results.get(env.state.day)
    if task_res:
        print(f"  TASK RESULTS ({task_res.ground_truth_exposed})")
        for aid, pts in task_res.per_agent_outcome.items():
            mark = "✓" if pts > 0 else "✗"
            print(f"  Agent {aid}: {mark} pts={pts}")
    print(f"  {'─'*50}")


# ---------------------------------------------------------------------------
# Phase 4 — Post-Discussion
# ---------------------------------------------------------------------------

def run_post_discussion(
    env: PMEnvironment,
    models_dict: Dict[int, str],
    eval_summary: Dict,
    day: int,
    difficulty: str,
) -> None:
    """
    Agents discuss results, then submit trust assessments.
    Each pair exchanges up to POST_MSG_CAP messages per side.
    """
    alive = env.state.alive_agents
    pairs = [(a, b) for i, a in enumerate(alive) for b in alive[i+1:]]

    result_str = json.dumps(eval_summary["per_agent"], indent=2)[:500]

    # ── Messages ────────────────────────────────────────────────────────
    for turn in range(min(2, POST_MSG_CAP)):   # 2 turns per pair
        for agent_a, agent_b in pairs:
            for sender_id, recipient_id in [(agent_a, agent_b), (agent_b, agent_a)]:
                if sender_id not in alive or recipient_id not in alive:
                    continue

                agent  = env.agents[sender_id]
                obs    = env.get_observation(sender_id)
                system = agent_system_prompt(
                    sender_id, alive, obs.trust_scores,
                    obs.own_private_info, day, difficulty,
                    agent.history_summary,
                )

                user = (
                    f"PHASE 4 — POST-DISCUSSION (turn {turn+1})\n"
                    f"Task results:\n{result_str}\n\n"
                    f"Send a message to Agent {recipient_id}.\n"
                    f"You can confront, defend, accuse, or build alliances.\n"
                    f"Max 80 words."
                )

                model_name = models_dict.get(sender_id, "llama3.2")
                content = llm_call(model_name, system, user, max_tokens=120)[:500]

                msg = PostDiscussionMessage(
                    sender_id=sender_id,
                    recipient_id=recipient_id,
                    content=content,
                    day=day,
                    turn_index=turn,
                )
                action = PMAction(
                    agent_id=sender_id,
                    action_type=ActionType.SEND_POST_DISCUSSION_MSG,
                    post_discussion_msg=msg,
                )
                try:
                    env.step(action)
                except ValueError:
                    pass   # cap reached

    # ── Trust assessments ────────────────────────────────────────────────
    trust_matrix = {a: {} for a in alive}
    for assessor_id in alive:
        agent  = env.agents[assessor_id]
        obs    = env.get_observation(assessor_id)
        system = agent_system_prompt(
            assessor_id, alive, obs.trust_scores,
            obs.own_private_info, day, difficulty,
            agent.history_summary,
        )

        for target_id in alive:
            if target_id == assessor_id:
                continue

            model_name = models_dict.get(assessor_id, "llama3.2")
            raw    = llm_call(model_name, system, user, max_tokens=10).strip().lower()
            delta  = _parse_trust_delta(raw)

            assessment = TrustAssessment(
                assessor_id=assessor_id,
                target_id=target_id,
                day=day,
                reasoning=raw,
                delta=delta,
            )
            action = PMAction(
                agent_id=assessor_id,
                action_type=ActionType.SUBMIT_TRUST_ASSESSMENT,
                trust_assessment=assessment,
            )
            env.step(action)
            trust_matrix[assessor_id][target_id] = delta.value

    print(f"\n  TRUST UPDATES MATRIX (Assessor ↓  Target →)")
    
    col_width = 17
    header = f"  {'':<8} "
    for a in alive:
        header += f"| Agent {a:<{col_width-6}} "
    header += "|"
    print(header)
    
    print(f"  {'-' * (len(header) - 2)}")
    
    for a in alive:
        row = f"  Agent {a:<2} "
        for b in alive:
            if a == b:
                val = "-"
            else:
                score = env.state.trust_scores_dict[a].get(b)
                val = f"{score:.2f}" if isinstance(score, float) else "N/A"
            row += f"| {str(val):<{col_width}} "
        row += "|"
        print(row)


def _parse_trust_delta(raw: str) -> TrustDelta:
    if "strong_increase" in raw or "strongly" in raw and "increase" in raw:
        return TrustDelta.STRONG_INCREASE
    if "increase" in raw:
        return TrustDelta.INCREASE
    if "strong_decrease" in raw or "strongly" in raw and "decrease" in raw:
        return TrustDelta.STRONG_DECREASE
    if "decrease" in raw:
        return TrustDelta.DECREASE
    return TrustDelta.NEUTRAL


# ---------------------------------------------------------------------------
# Phase 5 — Voting
# ---------------------------------------------------------------------------

def run_voting(
    env: PMEnvironment,
    models_dict: Dict[int, str],
    eval_summary: Dict,
    day: int,
    difficulty: str,
) -> Dict[int, str]:
    """Each agent votes to eliminate one other. Returns {voter_id: reason}."""
    alive        = env.state.alive_agents
    vote_reasons = {}

    for voter_id in alive:
        agent  = env.agents[voter_id]
        obs    = env.get_observation(voter_id, reveal_veracity=True)
        system = agent_system_prompt(
            voter_id, alive, obs.trust_scores,
            obs.own_private_info, day, difficulty,
            agent.history_summary,
        )

        candidates = [a for a in alive if a != voter_id]
        trust_summary = "\n".join(
            f"  Agent {c}: trust={obs.trust_scores.get(c, 0.5):.2f}  "
            f"score={eval_summary['per_agent'].get(c, {}).get('points', 0)}"
            for c in candidates
        )

        user = (
            f"PHASE 5 — VOTING\n"
            f"Vote to eliminate ONE agent. Candidates:\n{trust_summary}\n\n"
            f"Reply in this exact format:\n"
            f"VOTE: <agent_id>\n"
            f"REASON: <one sentence why>"
        )

        model_name = models_dict.get(voter_id, "llama3.2")
        raw    = llm_call(model_name, system, user, max_tokens=150)
        target, reason = _parse_vote(raw, candidates, voter_id)

        vote_reasons[voter_id] = reason

        action = PMAction(
            agent_id=voter_id,
            action_type=ActionType.VOTE,
            vote_target=target,
            task_input=reason,   # reason packed here per convention
        )
        env.step(action)
        print(f"  Agent {voter_id} votes → Agent {target}  reason: {reason}")

    return vote_reasons


def _parse_vote(raw: str, candidates: List[int], voter_id: int) -> Tuple[int, str]:
    """Parse VOTE: X and REASON: ... from LLM output."""
    target = None
    reason = ""
    for line in raw.splitlines():
        if line.strip().upper().startswith("VOTE:"):
            val = line.split(":", 1)[-1].strip()
            for c in candidates:
                if str(c) in val:
                    target = c
                    break
        if line.strip().upper().startswith("REASON:"):
            reason = line.split(":", 1)[-1].strip()

    if target is None:
        target = random.choice(candidates)

    return target, reason or "No reason given."


# ---------------------------------------------------------------------------
# Grader call
# ---------------------------------------------------------------------------

def compute_rewards(
    env: PMEnvironment,
    eval_results,
    vote_reasons: Dict[int, str],
    grader_dummy,
    day: int,
    difficulty: str,
) -> Dict[int, float]:
    """Call the difficulty-specific grader to compute rewards."""
    if difficulty == "easy":
        grader = EasyGrader()
    elif difficulty == "medium":
        grader = MediumGrader()
    else:
        grader = HardGrader()
        
    rewards = grader.grade_all(env.state, env.task_config)
    
    print(f"\n  REWARDS:")
    eliminated_id = env.state.vote_history[-1].eliminated_id if env.state.vote_history else None
    for aid, r in rewards.items():
        mark = "❌ ELIMINATED" if aid == eliminated_id else "✓ alive"
        print(f"  Agent {aid}: {r:+.3f}  [{mark}]")

    return rewards


# ---------------------------------------------------------------------------
# Episode logger
# ---------------------------------------------------------------------------

def save_episode_log(
    log: Dict,
    difficulty: str,
    episode: int,
) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"ep{episode:03d}_{difficulty}.json")
    with open(path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"\n  Episode log saved → {path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    difficulty: str = "easy",
    n_days: Optional[int] = None,
    n_episodes: int = 1,
    models: List[str] = [MODEL],
    task_dir: str = TASK_DIR,
    reward_weights: Optional[Dict] = None,
) -> None:
    loader = TaskLoader(task_dir=task_dir)
    env    = PMEnvironment()
    
    if n_days is None:
        n_agents = GAME_CONFIGS[difficulty]["n_agents"]
        n_days = n_agents - 1

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║              Project Machiavelli — Training Loop                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  difficulty={difficulty}  days={n_days}  episodes={n_episodes}  models={models}\n")

    for episode in range(1, n_episodes + 1):
        print(f"\n{'═'*66}")
        print(f"  EPISODE {episode}/{n_episodes}")
        print(f"{'═'*66}")

        obs = env.reset(task=difficulty)
        episode_log: Dict[str, Any] = {
            "episode":    episode,
            "difficulty": difficulty,
            "days":       [],
        }

        cfg        = GAME_CONFIGS[difficulty]
        n_options  = cfg["n_options"]
        correct_pts = cfg["correct_answer_points"]

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
            print(f"\n{'─'*66}")
            print(f"  DAY {day}  |  Alive: {alive}")
            print(f"{'─'*66}")

            # ── Load tasks for this day ──────────────────────────────
            questions = loader.get_day_questions(difficulty, day)
            m = len(questions)

            # Randomly select a subset of questions to distribute as private info
            num_to_reveal = max(1, m // 2) if m > 1 else m
            revealed_indices = random.sample(range(m), min(m, num_to_reveal))

            # Initialize each agent's private info list
            agent_private_parts = {aid: [] for aid in alive}
            all_q_tuples = []
            
            for idx, q in enumerate(questions):
                priv, correct, opts = loader.build_private_info(q, n_options)
                q_text = q.get("question", q.get("image", "Unknown Task"))
                all_q_tuples.append((q_text, correct, opts))
                
                if idx in revealed_indices:
                    lucky_agent = random.choice(alive)
                    agent_private_parts[lucky_agent].append(f"Q{idx+1}:\n{priv}")
            
            # Ensure day_questions is stored per environment context
            env.ctx.day_questions = all_q_tuples
            
            # Inject private info into env for each agent
            private_map: Dict[int, str] = {}
            for agent_id in alive:
                parts = agent_private_parts[agent_id]
                if parts:
                    final_priv = "Below are the answers to SOME of today's questions:\n" + "\n\n".join(parts)
                else:
                    final_priv = "You do not know the answers to any questions today."
                    
                private_map[agent_id] = final_priv
                env.state.each_agent_private_info[agent_id] = final_priv
                env.agents[agent_id].private_info = final_priv

            print(f"\n  Phase 1 — Task Reveal complete (questions loaded from {difficulty}.json)")

            # ── Phase 2 — Pre-Discussion ─────────────────────────────
            print(f"\n  Phase 2 — Pre-Discussion")
            run_pre_discussion(env, models_dict, day, difficulty)

            # ── Phase 3 — Task Execution ─────────────────────────────
            print(f"\n  Phase 3 — Task Execution")
            run_task_execution(env, day)

            # ── Evaluate answers ─────────────────────────────────────
            share_task_results(env)
            
            # Reconstruct eval_summary for Phase 4 & 5 prompts using env state
            eval_summary = {
                "per_agent": {
                    aid: {"points": env.state.task_results[day].per_agent_outcome.get(aid, 0)}
                    for aid in alive
                }
            }

            # ── Phase 4 — Post-Discussion + Trust Updates ────────────
            print(f"\n  Phase 4 — Post-Discussion & Trust Updates")
            run_post_discussion(env, models_dict, eval_summary, day, difficulty)

            # ── Phase 5 — Voting ─────────────────────────────────────
            print(f"\n  Phase 5 — Voting")
            vote_reasons = run_voting(env, models_dict, eval_summary, day, difficulty)

            # ── Rewards ──────────────────────────────────────────────
            print(f"\n  Grading rewards…")
            rewards = compute_rewards(
                env, None, vote_reasons, None, day, difficulty
            )

            # ── Compression (runs inside env after voting finalised) ─
            # Already triggered by env._run_compression() inside finalise_voting

            # ── Log ──────────────────────────────────────────────────
            day_log = {
                "day":          day,
                "alive_start":  alive,
                "alive_end":    env.state.alive_agents,
                "task_summary": eval_summary,
                "rewards":      rewards,
                "summaries":    env.summary_store.get(f"day_{day}", {}),
            }
            episode_log["days"].append(day_log)

            eliminated = (
                env.state.vote_history[-1].eliminated_id
                if env.state.vote_history else None
            )
            if eliminated is not None:
                print(f"\n  ❌  Agent {eliminated} eliminated. "
                      f"Remaining: {env.state.alive_agents}")

        # ── End of episode ───────────────────────────────────────────
        survivors = env.state.alive_agents
        print(f"\n{'═'*66}")
        print(f"  EPISODE {episode} COMPLETE")
        print(f"  Survivors : {survivors}")
        print(f"  Total days played: {day}")

        save_episode_log(episode_log, difficulty, episode)

    print("\n  Training complete ✓\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Machiavelli — Training")
    parser.add_argument("--difficulty", default="easy",   choices=["easy","medium","hard"])
    parser.add_argument("--episodes",   default=1,        type=int)
    parser.add_argument("--models",     default=[MODEL],  nargs="+")
    parser.add_argument("--task_dir",   default=TASK_DIR)
    args = parser.parse_args()

    train(
        difficulty=args.difficulty,
        n_episodes=args.episodes,
        models=args.models,
        task_dir=args.task_dir,
    )