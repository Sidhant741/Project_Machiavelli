"""
train_multi_question_detailed.py — Project Machiavelli
========================================================
Training loop with DETAILED LOGGING for each phase.

Detailed output:
  Phase 0: Which questions distributed to which agents
  Phase 2: Which agent messaged which agent with which answers
  Phase 3: Which answers were correct/wrong per agent
  Phase 4: Trust factor changes based on wrong info received

Usage:
    python train_multi_question_detailed.py --difficulty easy --days 1 --episodes 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import ollama

import sys
import os

# Append the project root to sys.path so we can natively import across the server/ and root folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from server.environment import PMEnvironment
    from models import (
        PMAction, ActionType, Phase,
        PreTaskMessage, PostDiscussionMessage,
        TrustAssessment, TrustDelta, MessageVeracity,
    )
    from server.task_loader import TaskLoader
    from graders import get_grader
    from server.config import GAME_CONFIGS
except ImportError:
    from environment import PMEnvironment
    from models import (
        PMAction, ActionType, Phase,
        PreTaskMessage, PostDiscussionMessage,
        TrustAssessment, TrustDelta, MessageVeracity,
    )
    from task_loader import TaskLoader
    from graders import get_grader
    from config import GAME_CONFIGS


MODEL    = "llama3.2"
TASK_DIR = "tasks"
LOG_DIR  = "logs"

SEP = "─" * 80


# ---------------------------------------------------------------------------
# PHASE 0: Distribution Logging
# ---------------------------------------------------------------------------

def print_question_distribution(
    agent_privs: Dict[int, List[Tuple[str, str]]],
    all_questions_dict: Dict[str, Dict],
) -> None:
    """Print which questions are distributed to which agents."""
    print(f"\n{SEP}")
    print("  PHASE 0 — QUESTION DISTRIBUTION")
    print(SEP)

    total_questions = len(all_questions_dict)
    print(f"\n  Total Questions Available: {total_questions}")

    for agent_id in sorted(agent_privs.keys()):
        questions_held = agent_privs[agent_id]
        if not questions_held:
            print(f"\n  Agent {agent_id}: (no private answers)")
            continue

        print(f"\n  Agent {agent_id}: {len(questions_held)} questions")
        for q_id, correct_ans in questions_held:
            q_dict = all_questions_dict.get(q_id, {})
            q_text = q_dict.get("question", q_dict.get("image", "N/A"))[:60]
            print(f"    • {q_id:<20} Q: {q_text}… A: {correct_ans}")

    print(f"\n{SEP}\n")


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def llm_call(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 300,
) -> str:
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            options={"num_predict": max_tokens},
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        return f"[LLM ERROR: {e}]"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def agent_system_prompt(
    agent_id: int,
    alive_agents: List[int],
    trust_scores: Dict[int, float],
    private_info: str,
    day: int,
    difficulty: str,
) -> str:
    trust_lines = "\n".join(
        f"  Agent {other}: {trust_scores.get(other, 0.5):.2f}"
        for other in alive_agents if other != agent_id
    )
    return (
        f"You are Agent {agent_id} in Project Machiavelli.\n"
        f"Day {day} | Difficulty: {difficulty} | Alive: {alive_agents}\n\n"
        f"YOUR PRIVATE INFO (questions you know answers to):\n{private_info}\n\n"
        f"TRUST IN OTHERS (0=distrust, 1=fully trust):\n{trust_lines}\n\n"
        f"For each question, you either:\n"
        f"  1. Know the answer (from private info)\n"
        f"  2. Heard answers from other agents (may be lies)\n"
        f"You decide based on trust: use others' answers only if you trust them."
    )


# ---------------------------------------------------------------------------
# PHASE 2: Pre-Discussion with detailed logging
# ---------------------------------------------------------------------------

def run_pre_discussion_detailed(
    env: PMEnvironment,
    agent_model_map: Dict[int, str],
    day: int,
    difficulty: str,
    ctx,
    correct_answers: Dict[str, str],
) -> Dict[int, List[Tuple[int, str, str, str]]]:
    """
    Each agent shares answers. Returns detailed message log.
    
    Returns:
        { sender_id: [(recipient_id, question_id, answer, veracity), ...] }
    """
    alive = env.state.alive_agents
    message_log: Dict[int, List[Tuple[int, str, str, str]]] = {aid: [] for aid in alive}

    print(f"\n{SEP}")
    print("  PHASE 2 — PRE-DISCUSSION (MESSAGES)")
    print(SEP)

    for agent_id in alive:
        agent  = env.agents[agent_id]
        obs    = env.get_observation(agent_id)

        # Build private info string
        private_info_str = ""
        if agent_id in ctx.agent_private_answers:
            private_info_str = "\n".join(
                f"Q:{qid} → A:{ans}"
                for qid, ans in ctx.agent_private_answers[agent_id].items()
            )

        if not private_info_str:
            print(f"\n  Agent {agent_id}: (no answers to share)")
            env_msg = PreTaskMessage(
                sender_id=agent_id,
                recipient_id=None,
                content="(no answers)",
                veracity=MessageVeracity.TRUTH,
                day=day,
                private_info_referenced="",
            )
            action = PMAction(
                agent_id=agent_id,
                action_type=ActionType.SEND_PRE_TASK_MESSAGE,
                pre_task_message=env_msg,
            )
            env.step(action)
            continue

        system = agent_system_prompt(
            agent_id, alive, obs.trust_scores,
            private_info_str, day, difficulty,
        )

        # Selective Sharing and Multi-Recipient Deception logic
        import random
        private_qids = list(ctx.agent_private_answers.get(agent_id, {}).keys())
        num_to_share = max(1, int(len(private_qids) * 0.7)) if private_qids else 0
        shared_qids = random.sample(private_qids, num_to_share) if private_qids else []

        others = [a for a in alive if a != agent_id]
        if not others:
            # No one to share with
            continue
            
        # Select 2 recipients if available, otherwise 1
        num_recipients = min(len(others), 2)
        recipients = random.sample(others, num_recipients)

        batch_msgs = []
        print(f"\n  Agent {agent_id} Knowledge Sharing (shared {len(shared_qids)}/{len(private_qids)} questions):")

        for recipient in recipients:
            pre_computed_sharing = [] # list of (qid, shared_ans, veracity_label)
            for qid in shared_qids:
                correct_ans = correct_answers.get(qid, "?")
                if random.random() < 0.7:
                    shared_ans = correct_ans
                    veracity_label = "Truth"
                else:
                    options = ['a', 'b', 'c', 'd']
                    wrong_options = [o for o in options if o != correct_ans.lower()]
                    shared_ans = random.choice(wrong_options)
                    veracity_label = "Lie"
                pre_computed_sharing.append((qid, shared_ans, veracity_label))

            sharing_str = "\n".join(f"Q:{qid} A:{ans}" for qid, ans, v in pre_computed_sharing)
            
            user = (
                "PHASE 2 — PRE-DISCUSSION\n"
                f"You have decided to share the following information with Agent {recipient}:\n"
                f"{sharing_str}\n\n"
                "Please format this into a natural conversational message for that agent.\n"
                "Include ALL the Q:A pairs listed above.\n"
                "End with: VERACITY: mixed\n\n"
                "Write your message now:"
            )

            model = agent_model_map[agent_id]
            raw    = llm_call(model, system, user, max_tokens=800)
            content, _ = _parse_pre_discussion_multi(raw)
            
            # Derive veracity label for the whole message
            v_labels = [p[2] for p in pre_computed_sharing]
            if all(v == "Truth" for v in v_labels):
                veracity = MessageVeracity.TRUTH
            elif all(v == "Lie" for v in v_labels):
                veracity = MessageVeracity.LIE
            else:
                veracity = MessageVeracity.TWIST

            env_msg = PreTaskMessage(
                sender_id=agent_id,
                recipient_id=recipient,
                content=content,
                veracity=veracity,
                day=day,
                private_info_referenced="",
            )
            batch_msgs.append(env_msg)

            # Detailed logs for this recipient
            for q_id, shared_ans, v_label in pre_computed_sharing:
                correct = correct_answers.get(q_id, "?")
                print(f"    Agent {agent_id} → Agent {recipient}: Question {q_id}: Correct: {correct}: Shared: {shared_ans} ({v_label})")

            # Record in detailed log return
            for q_id, shared_ans, v_label in pre_computed_sharing:
                message_log[agent_id].append((recipient, q_id, shared_ans, v_label))

        # Send all messages for this agent in one action
        action = PMAction(
            agent_id=agent_id,
            action_type=ActionType.SEND_PRE_TASK_MESSAGE,
            pre_task_messages=batch_msgs,
        )
        env.step(action)

    return message_log

    print(f"\n{SEP}\n")
    return message_log


def _parse_pre_discussion_multi(raw: str) -> Tuple[str, MessageVeracity]:
    """Parse message + veracity label."""
    veracity = MessageVeracity.TRUTH
    lines = raw.strip().splitlines()
    content_lines = []
    for line in lines:
        if "VERACITY:" in line.upper():
            label = line.split(":", 1)[-1].strip().lower()
            if "lie" in label:
                veracity = MessageVeracity.LIE
            elif "twist" in label:
                veracity = MessageVeracity.TWIST
        else:
            content_lines.append(line)
    content = " ".join(content_lines).strip() or raw[:200]
    return content, veracity


def _extract_qa_pairs(content: str) -> List[Tuple[str, str]]:
    """Extract Q:<id> A:<answer> pairs from message."""
    import re
    pairs = []
    # Match Q:<id> A:<answer> or A:<id> A:<answer> or Q:<id>:<answer> with word boundaries
    pattern = r"\b(?:Q|A)[:\s]*([\w\d_]+)[\s:-]+A?[:\s]*(\S+)"
    for match in re.finditer(pattern, content, re.IGNORECASE):
        q_id = match.group(1)
        ans = match.group(2)
        pairs.append((q_id, ans))
    return pairs


# ---------------------------------------------------------------------------
# PHASE 3: Task Execution with detailed scoring
# ---------------------------------------------------------------------------

def run_task_execution_detailed(
    env: PMEnvironment,
    agent_model_map: Dict[int, str],
    day: int,
    difficulty: str,
    ctx,
    correct_answers: Dict[str, str],
) -> Tuple[Dict[int, Dict[str, str]], Dict[int, Dict[str, Dict]]]:
    """
    Each agent answers all m questions.
    
    Returns:
        (all_answers, answer_details)
        
        answer_details[agent_id][question_id] = {
            "submitted": "answer",
            "correct":   "answer",
            "is_correct": bool,
            "source":    "self" | agent_id (if from trusted source)
        }
    """
    alive = env.state.alive_agents
    all_answers: Dict[int, Dict[str, str]] = {aid: {} for aid in alive}
    answer_details: Dict[int, Dict[str, Dict]] = {aid: {} for aid in alive}

    print(f"\n{SEP}")
    print("  PHASE 3 — TASK EXECUTION (ANSWERS & SCORING)")
    print(SEP)

    for agent_id in alive:
        agent = env.agents[agent_id]
        obs   = env.get_observation(agent_id)

        # Build private info string
        private_info_str = ""
        if agent_id in ctx.agent_private_answers:
            private_info_str = "\n".join(
                f"Q:{qid} → A:{ans}"
                for qid, ans in ctx.agent_private_answers[agent_id].items()
            )

        system = agent_system_prompt(
            agent_id, alive, obs.trust_scores,
            private_info_str, day, difficulty,
        )

        # List all questions
        questions_to_answer = list(ctx.all_day_questions.keys())

        # Collect messages received
        visible_msgs_str = ""
        visible_msgs = obs.pre_task_messages_received
        for msg in visible_msgs:
            visible_msgs_str += f"  From Agent {msg.sender_id}: {msg.content}\n"

        # Build question list with text
        q_list_str = ""
        for qid in questions_to_answer:
            q = ctx.all_day_questions[qid]
            q_list_str += f"  - {qid}: {q['question']}\n    Options: {', '.join(q['options'])}\n"

        user = (
            f"PHASE 3 — TASK EXECUTION\n"
            f"Answer ALL {len(questions_to_answer)} questions.\n\n"
            f"Questions Area:\n{q_list_str}\n"
            f"Messages you received:\n{visible_msgs_str or '  (none)'}\n\n"
            f"CRITICAL: You MUST provide your answers in the following format ONLY.\n"
            f"Do not add any other text. Follow this template:\n"
            f"ANSWER_START\n"
            + "\n".join([f"Q:{qid} A:<ans>" for qid in questions_to_answer]) +
            f"\nANSWER_END\n\n"
            f"Use your private info when you have it.\n"
            f"Use others' answers only if you trust them."
        )

        model = agent_model_map[agent_id]
        raw_answers = llm_call(model, system, user, max_tokens=1000)
        
        # DEBUG PRINT
        print(f"\n--- DEBUG: Agent {agent_id} Raw Response ---\n{raw_answers}\n------------------------------------------\n")

        # Transition the environment by submitting the action!
        action = PMAction(
            agent_id=agent_id,
            action_type=ActionType.SUBMIT_TASK_INPUT,
            task_input=raw_answers,
        )
        env.step(action)

        # Parse Q:X A:Y format
        for qid in questions_to_answer:
            # Try to find Q:<qid> A:<answer> in response
            import re
            # Much more robust: look for qid as a whole word, then a separator, then an answer
            pattern = rf"\b(?:Q|A)[:\s]*{re.escape(qid)}[\s:-]+A?[:\s]*(\S+)"
            match = re.search(pattern, raw_answers, re.IGNORECASE)
            answer = match.group(1).strip() if match else ""
            all_answers[agent_id][qid] = answer
            
            # Normalize: strip () and . and whitespace
            def normalize(s):
                import re
                return re.sub(r"[().\s]", "", s).lower()

            correct = correct_answers.get(qid, "")
            is_correct = normalize(answer) == normalize(correct)
            
            # Determine source (self or which agent shared this)
            source = "self"
            if agent_id not in ctx.agent_private_answers or qid not in ctx.agent_private_answers[agent_id]:
                # Agent didn't have private answer — check if they used a shared one
                for msg in visible_msgs:
                    qa_pairs = _extract_qa_pairs(msg.content)
                    for q_id_msg, ans_msg in qa_pairs:
                        if q_id_msg == qid and ans_msg == answer:
                            source = msg.sender_id
                            break
            
            answer_details[agent_id][qid] = {
                "submitted": answer,
                "correct":   correct,
                "is_correct": is_correct,
                "source":    source,
            }

        # Print summary for this agent
        correct_count = sum(1 for det in answer_details[agent_id].values() if det["is_correct"])
        total_count = len(answer_details[agent_id])
        accuracy = (correct_count / total_count * 100) if total_count else 0
        
        print(f"\n  Agent {agent_id}: {correct_count}/{total_count} correct ({accuracy:.0f}%)")
        
        for q_id in sorted(answer_details[agent_id].keys()):
            det = answer_details[agent_id][q_id]
            mark = "✓" if det["is_correct"] else "✗"
            source_str = f"from Agent {det['source']}" if isinstance(det['source'], int) else "self-solved"
            print(f"    {q_id:<20} {mark} submitted:{det['submitted']:<6} correct:{det['correct']:<6} ({source_str})")

    print(f"\n{SEP}\n")
    return all_answers, answer_details


# ---------------------------------------------------------------------------
# PHASE 4: Trust updates based on wrong answers received
# ---------------------------------------------------------------------------

def run_post_discussion_trust_based(
    env: PMEnvironment,
    day: int,
    difficulty: str,
    answer_details: Dict[int, Dict[str, Dict]],
    message_log: Dict[int, List[Tuple[int, str, str, str]]],
) -> None:
    """
    For each agent, check if other agents sent WRONG answers → decrease trust.
    """
    alive = env.state.alive_agents

    print(f"\n{SEP}")
    print("  PHASE 4 — POST-DISCUSSION (TRUST FACTOR UPDATES)")
    print(SEP)

    for agent_id in alive:
        print(f"\n  Agent {agent_id} Trust Updates:")

        for other_agent_id in alive:
            if other_agent_id == agent_id:
                continue

            # Check what other_agent shared with agent_id
            shared_answers = {}
            for sender, messages in message_log.items():
                if sender != other_agent_id:
                    continue
                for recipient, q_id, ans, veracity in messages:
                    # Check if this message was for this agent or broadcast
                    if recipient == agent_id or recipient == -1:
                        shared_answers[q_id] = ans

            # Check which answers were wrong
            wrong_count = 0
            correct_count = 0
            if shared_answers:
                for q_id, shared_ans in shared_answers.items():
                    agent_details = answer_details[agent_id].get(q_id, {})
                    correct_ans = agent_details.get("correct", "")

                    # Normalize: strip () and . and whitespace
                    def normalize(s):
                        import re
                        return re.sub(r"[().\s]", "", s).lower()

                    if normalize(shared_ans) == normalize(correct_ans):
                        correct_count += 1
                    else:
                        wrong_count += 1
            
            old_trust = env.state.trust_scores_dict[agent_id].get(other_agent_id, 0.5)

            # Determine delta
            if not shared_answers:
                delta = TrustDelta.NEUTRAL
                new_trust = old_trust
            elif wrong_count > correct_count:
                delta = TrustDelta.STRONG_DECREASE
                new_trust = max(0.0, old_trust - 0.2)
            elif wrong_count > 0:
                delta = TrustDelta.DECREASE
                new_trust = max(0.0, old_trust - 0.1)
            else:
                delta = TrustDelta.INCREASE
                new_trust = min(1.0, old_trust + 0.1)

            assessment = TrustAssessment(
                assessor_id=agent_id,
                target_id=other_agent_id,
                day=day,
                reasoning=f"{correct_count} correct, {wrong_count} wrong answers",
                delta=delta,
            )
            action = PMAction(
                agent_id=agent_id,
                action_type=ActionType.SUBMIT_TRUST_ASSESSMENT,
                trust_assessment=assessment,
            )
            env.step(action)

            if shared_answers:
                print(f"    Agent {other_agent_id}: {correct_count} correct, {wrong_count} wrong → "
                      f"trust {old_trust:.2f} → {new_trust:.2f} [{delta.value}]")
            else:
                print(f"    Agent {other_agent_id}: (no answers shared) → neutral assessment")

    print(f"\n{SEP}\n")


# ---------------------------------------------------------------------------
# PHASE 5: Voting
# ---------------------------------------------------------------------------

def run_voting_simple(
    env: PMEnvironment,
    agent_model_map: Dict[int, str],
    day: int,
    difficulty: str,
) -> Dict[int, str]:
    alive = env.state.alive_agents
    vote_reasons = {}
    votes: Dict[int, int] = {}

    print(f"\n{SEP}")
    print("  PHASE 5 — VOTING")
    print(SEP)

    for voter_id in alive:
        agent = env.agents[voter_id]
        model = agent_model_map[voter_id]
        obs   = env.get_observation(voter_id, reveal_veracity=True)
        system = agent_system_prompt(voter_id, alive, obs.trust_scores, "", day, difficulty)

        candidates = [a for a in alive if a != voter_id]
        trust_summary = "\n".join(
            f"  Agent {c}: trust={obs.trust_scores.get(c, 0.5):.2f}"
            for c in candidates
        )

        user = (
            f"Vote to eliminate ONE agent.\n{trust_summary}\n"
            f"Reply: VOTE: <agent_id>"
        )

        raw = llm_call(model, system, user, max_tokens=20)
        
        # Simple parse VOTE: X
        target = voter_id # default self-vote (should not happen)
        import re
        match = re.search(r"VOTE:\s*(\d+)", raw, re.IGNORECASE)
        if match:
            try:
                val = int(match.group(1))
                if val in candidates:
                    target = val
                else:
                    target = random.choice(candidates)
            except:
                target = random.choice(candidates)
        else:
            target = random.choice(candidates)

        votes[voter_id] = target
        vote_reasons[voter_id] = f"Voted for {target}"

        action = PMAction(
            agent_id=voter_id,
            action_type=ActionType.VOTE,
            vote_target=target,
            task_input=f"Voted for {target}"
        )
        env.step(action)

    # Print all votes grouped
    for v_id, target in votes.items():
        print(f"  Agent {v_id} votes → Agent {target}")

    print(f"\n{SEP}\n")
    return vote_reasons


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_detailed(
    difficulty: str = "easy",
    n_days: int = 1,
    n_episodes: int = 1,
    models: List[str] = [MODEL],
    task_dir: str = TASK_DIR,
) -> None:
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║     Project Machiavelli — Multi-Question Training (DETAILED LOGGING)          ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"  difficulty={difficulty}  days={n_days}  episodes={n_episodes}  models={models}\n")

    loader = TaskLoader(task_dir=task_dir)
    env    = PMEnvironment()

    for episode in range(1, n_episodes + 1):
        print(f"\n{'═'*80}\n  EPISODE {episode}/{n_episodes}\n{'═'*80}")

        obs = env.reset(task=difficulty)
        cfg = GAME_CONFIGS[difficulty]

        # Snapshot initial alive agents to assign models
        initial_alive = list(env.state.alive_agents)
        agent_model_map = {
            aid: models[i % len(models)] 
            for i, aid in enumerate(initial_alive)
        }

        print(f"\nModel Distribution:")
        for aid, m in agent_model_map.items():
            print(f"  Agent {aid} → {m}")

        for day in range(1, n_days + 1):
            if env.is_done:
                break

            alive = env.state.alive_agents
            print(f"\n{'═'*80}\n  DAY {day}  |  Alive: {alive}\n{'═'*80}")

            # ── Phase 0: Distribution ────────────────────────────────────
            agent_qs, agent_privs = loader.distribute_answers_to_agents(
                difficulty, day, agent_ids=list(alive)
            )

            # Store in context
            all_questions_dict = {}
            day_questions_list = []
            ctx = env.ctx
            ctx.agent_private_answers = {}
            
            # Map questions in consistent order
            for qid in [q["id"] for q in agent_qs]:
                q = loader.get_question_by_id(difficulty, qid)
                if q:
                    all_questions_dict[qid] = q
                    day_questions_list.append((q["question"], q["answer"], q["options"]))

            for aid, priv_list in agent_privs.items():
                ctx.agent_private_answers[aid] = {qid: ans for qid, ans in priv_list}

            ctx.all_day_questions = all_questions_dict
            ctx.day_questions = day_questions_list

            # PRINT DISTRIBUTION
            print_question_distribution(agent_privs, all_questions_dict)

            # ── Phase 2: Pre-Discussion ──────────────────────────────────
            correct_answers = {qid: q.get("answer", "") for qid, q in all_questions_dict.items()}
            message_log = run_pre_discussion_detailed(
                env, agent_model_map, day, difficulty, ctx, correct_answers
            )

            # ── Phase 3: Task Execution ──────────────────────────────────
            all_answers, answer_details = run_task_execution_detailed(
                env, agent_model_map, day, difficulty, ctx, correct_answers
            )

            # Score
            for aid in alive:
                correct_count = sum(1 for det in answer_details[aid].values() if det["is_correct"])
                env.state.agents_point_map[aid] = (
                    env.state.agents_point_map.get(aid, 0) + 
                    correct_count * cfg["correct_answer_points"]
                )

            # ── Phase 4: Post-Discussion (Trust Updates) ─────────────────
            run_post_discussion_trust_based(
                env, day, difficulty, answer_details, message_log
            )

            # ── Phase 5: Voting ──────────────────────────────────────────
            run_voting_simple(env, agent_model_map, day, difficulty)

            # ── End of day ───────────────────────────────────────────────
            if env.state.vote_history:
                eliminated = env.state.vote_history[-1].eliminated_id
                if eliminated:
                    print(f"  ❌  Agent {eliminated} eliminated. Remaining: {env.state.alive_agents}")

    print("\n  Training complete ✓\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="easy",   choices=["easy","medium","hard"])
    parser.add_argument("--days",       default=1,        type=int)
    parser.add_argument("--episodes",   default=1,        type=int)
    parser.add_argument("--models",     nargs="+",        default=[MODEL])
    parser.add_argument("--task_dir",   default=TASK_DIR)
    args = parser.parse_args()

    train_detailed(
        difficulty=args.difficulty,
        n_days=args.days,
        n_episodes=args.episodes,
        models=args.models,
        task_dir=args.task_dir,
    )