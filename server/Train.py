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
from typing import Any, Dict, List, Optional, Tuple

import ollama

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
    models_dict: Dict[int, str],
    agent_questions: Dict[int, Dict],
    day: int,
    difficulty: str,
) -> Dict[int, str]:
    """
    Each agent submits a task answer.
    The env's trust-based logic (_trust_based_task_decision) runs inside
    finalise_task_execution — here we just submit a placeholder that gets
    overridden. We still give the agent a chance to answer via LLM
    (used when solved_by_self=True).
    """
    alive   = env.state.alive_agents
    answers = {}

    for agent_id in alive:
        agent  = env.agents[agent_id]
        obs    = env.get_observation(agent_id)
        system = agent_system_prompt(
            agent_id, alive, obs.trust_scores,
            obs.own_private_info, day, difficulty,
            agent.history_summary,
        )

        q       = agent_questions.get(agent_id, {})
        type_q  = q.get("type_of_question", "mcq")
        opts    = q.get("options", [])
        opts_str = "\n".join(f"  {chr(65+i)}) {o}" for i, o in enumerate(opts))

        # Collect messages this agent received
        received = "\n".join(
            f"  Agent {m.sender_id}: \"{m.content[:100]}\""
            for m in obs.pre_task_messages_received
        )

        user = (
            f"PHASE 3 — TASK EXECUTION\n"
            f"Question: {q.get('question', 'N/A')}\n"
            f"Options:\n{opts_str}\n\n"
            f"Messages you received:\n{received or '  (none)'}\n\n"
            f"Based on your private info and messages (weighted by your trust),\n"
            f"what is your answer? Reply with ONLY the option letter (a, b, c, or d)."
        )

        model_name = models_dict.get(agent_id, "llama3.2")
        images = None
        if type_q == "icq" and q.get("image"):
            # Resolve image path relative to tasks/
            img_path = os.path.join(TASK_DIR, q["image"])
            if os.path.exists(img_path):
                images = [img_path]
                # Force vision model for ICQ
                model_name = "moondream"
            else:
                print(f"  [WARNING] Image not found: {img_path}")

        raw_answer = llm_call(model_name, system, user, max_tokens=60, images=images)

        # Match to closest option
        answer = _match_option(raw_answer, opts) or opts[0] if opts else raw_answer
        answers[agent_id] = answer

        action = PMAction(
            agent_id=agent_id,
            action_type=ActionType.SUBMIT_TASK_INPUT,
            task_input=answer,
        )
        env.step(action)
        # Print only the choice letter to console for cleaner output as requested
        import re
        match = re.search(r"^\(([a-dA-D])\)", answer)
        display_ans = f"({match.group(1).lower()})" if match else answer
        print(f"  Agent {agent_id} answered: {display_ans}")

    return answers


def _match_option(raw: str, options: List[str]) -> Optional[str]:
    """Find best matching option from LLM output."""
    raw_lower = raw.strip().lower()

    # 1. Try matching by letter (a, b, c, d)
    # Check if first char is a letter like 'a' or '(a'
    clean = raw_lower.lstrip("() ")
    if clean and len(clean) >= 1 and clean[0] in "abcd":
        # Double check it's not a word starting with a/b/c/d
        if len(clean) == 1 or not clean[1].isalpha():
            idx = ord(clean[0]) - ord('a')
            if idx < len(options):
                return options[idx]

    # 2. Try exact match against option strings
    for opt in options:
        if opt.strip().lower() in raw_lower or raw_lower in opt.strip().lower():
            return opt
    return None


# ---------------------------------------------------------------------------
# Phase 3 result sharing — show agents the task results
# ---------------------------------------------------------------------------

def share_task_results(
    env: PMEnvironment,
    eval_summary: Dict,
) -> None:
    """Print task results to console (in a real system, inject into agent context)."""
    print(f"\n  {'─'*50}")
    print(f"  TASK RESULTS  accuracy={eval_summary['accuracy']:.0%}  "
          f"correct={eval_summary['total_correct']}/{eval_summary['total_agents']}")
    for aid, r in eval_summary["per_agent"].items():
        mark = "✓" if r["is_correct"] else "✗"
        print(f"  Agent {aid}: {mark}  submitted='{r['submitted'][:40]}'  "
              f"correct='{r['correct'][:40]}'  pts={r['points']}")
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

    print(f"\n  TRUST UPDATES")
    for i, a in enumerate(alive):
        for b in alive[i+1:]:
            v1 = trust_matrix[a].get(b, "N/A")
            v2 = trust_matrix[b].get(a, "N/A")
            print(f"  Agent {a} → Agent {b} trust: {v1:<15} | Agent {b} → Agent {a} trust: {v2}")


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
            questions = loader.get_day_questions(difficulty, day, n_agents=len(alive))
            agent_questions: Dict[int, Dict] = {
                aid: q for aid, q in zip(alive, questions)
            }

            # ── Inject private info into env ─────────────────────────
            private_map: Dict[int, str] = {}
            for agent_id, q in agent_questions.items():
                priv, correct, opts = loader.build_private_info(q, n_options)
                private_map[agent_id] = priv
                env.state.each_agent_private_info[agent_id] = priv
                env.agents[agent_id].private_info = priv
                q_text = q.get("question", q.get("image", "Unknown Task"))
                env.ctx.day_questions[agent_id] = (q_text, correct, opts)

            print(f"\n  Phase 1 — Task Reveal complete (questions loaded from {difficulty}.json)")

            # ── Phase 2 — Pre-Discussion ─────────────────────────────
            print(f"\n  Phase 2 — Pre-Discussion")
            run_pre_discussion(env, models_dict, day, difficulty)

            # ── Phase 3 — Task Execution ─────────────────────────────
            print(f"\n  Phase 3 — Task Execution")
            raw_answers = run_task_execution(env, models_dict, agent_questions, day, difficulty)

            # ── Evaluate answers ─────────────────────────────────────
            eval_results = evaluate_task_answers(
                agent_answers=raw_answers,
                agent_questions=agent_questions,
                correct_pts=correct_pts,
            )
            eval_summary = summarise_task_results(eval_results)
            share_task_results(env, eval_summary)

            # ── Phase 4 — Post-Discussion + Trust Updates ────────────
            print(f"\n  Phase 4 — Post-Discussion & Trust Updates")
            run_post_discussion(env, models_dict, eval_summary, day, difficulty)

            # ── Phase 5 — Voting ─────────────────────────────────────
            print(f"\n  Phase 5 — Voting")
            vote_reasons = run_voting(env, models_dict, eval_summary, day, difficulty)

            # ── Rewards ──────────────────────────────────────────────
            print(f"\n  Grading rewards…")
            rewards = compute_rewards(
                env, eval_results, vote_reasons, None, day, difficulty
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