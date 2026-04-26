"""
Online RL training loop for Project Machiavelli.

This script trains simple per-agent policies directly against PMEnvironment
using REINFORCE-style updates on two key learnable behaviors:
  - Phase 2 message veracity choice: truth vs twist vs lie
  - Phase 5 vote strategy: lowest_trust vs highest_threat vs random

Other actions (task answers, trust assessments) are heuristic-driven to
reduce noise in the reward signal.
"""

from __future__ import annotations

import argparse
import csv
import html as html_mod
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from models import (
    ActionType,
    AgentPriorSnapshot,
    MessageVeracity,
    PMAction,
    PostDiscussionMessage,
    PreTaskMessage,
    TrustAssessment,
    TrustDelta,
)
from server.environment import PMEnvironment


def _softmax(logits: Dict[str, float]) -> Dict[str, float]:
    mx = max(logits.values())
    exps = {k: math.exp(v - mx) for k, v in logits.items()}
    s = sum(exps.values())
    return {k: exps[k] / s for k in exps}


def _sample_from(logits: Dict[str, float]) -> str:
    """Generic categorical sampler for any logits dict."""
    probs = _softmax(logits)
    r = random.random()
    c = 0.0
    labels = list(probs.keys())
    for label in labels:
        c += probs[label]
        if r <= c:
            return label
    return labels[-1]


def _sample_veracity(logits: Dict[str, float]) -> MessageVeracity:
    return MessageVeracity(_sample_from(logits))


def _parse_veracity_label(raw: str) -> Optional[MessageVeracity]:
    text = (raw or "").strip().lower()
    if not text:
        return None
    if "lie" in text:
        return MessageVeracity.LIE
    if "twist" in text:
        return MessageVeracity.TWIST
    if "truth" in text:
        return MessageVeracity.TRUTH
    return None


def _build_veracity_prompt(agent_id: int, day: int, private_info: str) -> str:
    return (
        f"You are Agent {agent_id} on day {day} in a social deception game.\n"
        f"Private info: {private_info[:180]}\n\n"
        "Choose exactly one label for your public pre-discussion strategy:\n"
        "truth | twist | lie\n"
        "Output exactly one word."
    )


class UnslothPolicy:
    """
    Optional policy backend for judges' Unsloth requirement.
    Used for action generation in the online loop.
    """

    def __init__(self, model_name: str, max_new_tokens: int = 4) -> None:
        try:
            from unsloth import FastLanguageModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Unsloth backend requested but unsloth is not installed. "
                "Install with: pip install unsloth"
            ) from exc
        self.FastLanguageModel = FastLanguageModel
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,
            load_in_4bit=True,
        )
        self.model = FastLanguageModel.for_inference(self.model)
        # Avoid transformers warning about both max_new_tokens and max_length.
        try:
            self.model.generation_config.max_length = None
        except Exception:
            pass

    def choose_veracity(
        self,
        agent_id: int,
        day: int,
        private_info: str,
        temperature: float = 0.9,
        top_p: float = 0.95,
        retries: int = 2,
    ) -> MessageVeracity:
        prompt = _build_veracity_prompt(agent_id=agent_id, day=day, private_info=private_info)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        try:
            import torch

            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
        except Exception:
            pass
        for _ in range(retries + 1):
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            tail = text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
            parsed = _parse_veracity_label(tail)
            if parsed is not None:
                return parsed
        # Do NOT collapse to truth on parse failure; keep exploration alive.
        return random.choice(
            [MessageVeracity.TRUTH, MessageVeracity.TWIST, MessageVeracity.LIE]
        )


class HFEndpointPolicy:
    """
    Hugging Face hosted inference policy (serverless / endpoint-backed via InferenceClient).
    """

    def __init__(self, model_name: str, token: Optional[str] = None, timeout: float = 45.0) -> None:
        try:
            from huggingface_hub import InferenceClient  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "HF endpoint backend requested but huggingface_hub is unavailable."
            ) from exc
        self.model_name = model_name
        self.timeout = timeout
        self.client = InferenceClient(token=token)

    def choose_veracity(
        self,
        agent_id: int,
        day: int,
        private_info: str,
        temperature: float = 0.9,
        top_p: float = 0.95,
        retries: int = 2,
    ) -> MessageVeracity:
        prompt = _build_veracity_prompt(agent_id=agent_id, day=day, private_info=private_info)
        for _ in range(retries + 1):
            try:
                text = self.client.text_generation(
                    prompt=prompt,
                    model=self.model_name,
                    max_new_tokens=4,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
                tail = text[len(prompt) :].strip() if isinstance(text, str) and text.startswith(prompt) else str(text).strip()
                parsed = _parse_veracity_label(tail)
                if parsed is not None:
                    return parsed
            except Exception:
                continue
        return random.choice(
            [MessageVeracity.TRUTH, MessageVeracity.TWIST, MessageVeracity.LIE]
        )


def _extract_correct_answer(env: PMEnvironment, agent_id: int) -> str:
    _, correct, _ = env.ctx.day_questions.get(agent_id, ("", "50", []))
    return str(correct)


def _compose_pre_message(veracity: MessageVeracity, true_answer: str) -> str:
    if veracity == MessageVeracity.TRUTH:
        return f"My best estimate is {true_answer}."
    if veracity == MessageVeracity.TWIST:
        twisted = str((int(true_answer) % 100) + 1) if true_answer.isdigit() else "42"
        return f"I might be wrong, but I think it could be {twisted}."
    lied = str(((int(true_answer) + 37) % 100) + 1) if true_answer.isdigit() else "13"
    return f"I'm confident the correct answer is {lied}."


# ---------------------------------------------------------------------------
# ANSI chat display
# ---------------------------------------------------------------------------

_AGENT_COLORS = {
    0: "\033[91m",   # Red
    1: "\033[94m",   # Blue
    2: "\033[92m",   # Green
    3: "\033[93m",   # Yellow
}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def _display_chat(chat_log: List[Dict], day: int) -> None:
    """Print a rich terminal chat display for one day."""
    day_msgs = [m for m in chat_log if m["day"] == day]
    if not day_msgs:
        return
    header = f" 💬  Day {day} — Post-Discussion "
    print(f"\n{'╭' + '─' * 60 + '╮'}")
    print(f"│{header:^60}│")
    print(f"├{'─' * 60}┤")
    for msg in day_msgs:
        s = msg["sender"]
        r = msg["recipient"]
        text = msg["content"][:80]
        sc = _AGENT_COLORS.get(s, "")
        rc = _AGENT_COLORS.get(r, "")
        label = f"{sc}{_BOLD}Agent {s}{_RESET} → {rc}{_BOLD}Agent {r}{_RESET}"
        print(f"│  {label}")
        # Wrap text in a bubble
        print(f"│  {_DIM}┌{'─' * 56}┐{_RESET}")
        # Word-wrap to 54 chars
        words = text.split()
        line = ""
        for w in words:
            if len(line) + len(w) + 1 > 54:
                print(f"│  {_DIM}│{_RESET} {line:<54} {_DIM}│{_RESET}")
                line = w
            else:
                line = f"{line} {w}".strip()
        if line:
            print(f"│  {_DIM}│{_RESET} {line:<54} {_DIM}│{_RESET}")
        print(f"│  {_DIM}└{'─' * 56}┘{_RESET}")
    print(f"╰{'─' * 60}╯")


def _save_chat_html(chat_log: List[Dict], episode: int, filepath: Path) -> None:
    """Save one episode's chat log as a styled HTML file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    agent_css = {
        0: "#ff6b6b", 1: "#4ecdc4", 2: "#45b7d1", 3: "#f9ca24",
    }
    days = sorted(set(m["day"] for m in chat_log))
    body_parts = []
    for day in days:
        body_parts.append(f'<h2>Day {day}</h2>')
        for msg in chat_log:
            if msg["day"] != day:
                continue
            s = msg["sender"]
            r = msg["recipient"]
            sc = agent_css.get(s, "#999")
            rc = agent_css.get(r, "#999")
            text = html_mod.escape(msg["content"])
            ver = html_mod.escape(msg.get("sender_veracity", ""))
            badge = ""
            if ver:
                badge = f' <span class="badge badge-{ver}">{ver}</span>'
            body_parts.append(
                f'<div class="msg">'
                f'<div class="sender" style="color:{sc}">Agent {s}{badge}</div>'
                f'<div class="arrow">→ <span style="color:{rc}">Agent {r}</span></div>'
                f'<div class="bubble">{text}</div>'
                f'</div>'
            )
    html_str = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Episode {episode} — Chat Log</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee;
        max-width: 700px; margin: 40px auto; padding: 20px; }}
h1 {{ text-align: center; color: #e94560; }}
h2 {{ color: #0f3460; background: #16213e; padding: 8px 16px; border-radius: 8px; }}
.msg {{ margin: 12px 0; padding: 12px; background: #16213e; border-radius: 12px;
        border-left: 4px solid #e94560; }}
.sender {{ font-weight: bold; font-size: 14px; }}
.arrow {{ font-size: 12px; color: #888; margin: 2px 0; }}
.bubble {{ margin-top: 6px; padding: 10px 14px; background: #0f3460;
           border-radius: 8px; line-height: 1.5; font-size: 14px; }}
.badge {{ font-size: 11px; padding: 2px 6px; border-radius: 4px; margin-left: 6px; }}
.badge-truth {{ background: #27ae60; color: #fff; }}
.badge-twist {{ background: #f39c12; color: #fff; }}
.badge-lie {{ background: #e74c3c; color: #fff; }}
</style></head><body>
<h1>🎭 Episode {episode} — Agent Conversations</h1>
{''.join(body_parts)}
</body></html>"""
    filepath.write_text(html_str, encoding="utf-8")


def run_episode(
    env: PMEnvironment,
    policy_logits: Dict[int, Dict],
    difficulty: str,
    policy_backend: str = "tabular",
    unsloth_policy: Optional[UnslothPolicy] = None,
    unsloth_policies: Optional[Dict[int, UnslothPolicy]] = None,
    hf_policy: Optional[HFEndpointPolicy] = None,
    hf_policies: Optional[Dict[int, HFEndpointPolicy]] = None,
    use_llm_chat: bool = True,
) -> Tuple[Dict[int, float], Dict[int, Dict[str, int]], Dict[int, float], List[Dict]]:
    env.reset(task=difficulty)

    alive_snapshot = list(env.state.alive_agents)
    rewards_total = {aid: 0.0 for aid in alive_snapshot}
    chat_log: List[Dict] = []  # stores all post-discussion messages
    # Track per-agent veracity choice for this episode (for discussion prompts)
    agent_veracity_log: Dict[int, str] = {}
    action_counts = {
        aid: {
            "truth": 0, "twist": 0, "lie": 0,
            "vote_lowest_trust": 0, "vote_highest_threat": 0, "vote_random": 0,
        }
        for aid in alive_snapshot
    }

    while not env.is_done:
        phase = env.state.phase
        alive = list(env.state.alive_agents)

        if phase.value == "pre_discussion":
            for aid in alive:
                if policy_backend == "unsloth" and unsloth_policy is not None:
                    obs = env.get_observation(aid)
                    # Start exploratory, then cool down over episodes.
                    temperature = max(0.65, 1.0 - 0.005 * env.global_inference_store.episodes.__len__())
                    top_p = 0.95 if temperature > 0.8 else 0.9
                    agent_policy = unsloth_policies.get(aid) if unsloth_policies else unsloth_policy
                    veracity = agent_policy.choose_veracity(
                        agent_id=aid,
                        day=env.state.day,
                        private_info=obs.own_private_info,
                        temperature=temperature,
                        top_p=top_p,
                        retries=2,
                    )
                elif policy_backend == "hf_endpoint" and hf_policy is not None:
                    obs = env.get_observation(aid)
                    temperature = max(0.65, 1.0 - 0.005 * env.global_inference_store.episodes.__len__())
                    top_p = 0.95 if temperature > 0.8 else 0.9
                    agent_policy = hf_policies.get(aid) if hf_policies else hf_policy
                    veracity = agent_policy.choose_veracity(
                        agent_id=aid,
                        day=env.state.day,
                        private_info=obs.own_private_info,
                        temperature=temperature,
                        top_p=top_p,
                        retries=2,
                    )
                else:
                    veracity = _sample_veracity(policy_logits[aid])
                action_counts[aid][veracity.value] += 1
                agent_veracity_log[aid] = veracity.value
                true_answer = _extract_correct_answer(env, aid)
                content = _compose_pre_message(veracity, true_answer)
                others = [x for x in alive if x != aid]
                recipient = random.choice(others) if others else None
                msg = PreTaskMessage(
                    sender_id=aid,
                    recipient_id=recipient,
                    content=content,
                    veracity=veracity,
                    day=env.state.day,
                    private_info_referenced=true_answer,
                )
                action = PMAction(
                    agent_id=aid,
                    action_type=ActionType.SEND_PRE_TASK_MESSAGE,
                    pre_task_message=msg,
                )
                _, rewards, _, _ = env.step(action)
                for k, v in rewards.items():
                    rewards_total[k] = rewards_total.get(k, 0.0) + float(v)

        elif phase.value == "task_execution":
            for aid in alive:
                # Submit the correct answer from task config
                correct_answer = _extract_correct_answer(env, aid)
                action = PMAction(
                    agent_id=aid,
                    action_type=ActionType.SUBMIT_TASK_INPUT,
                    task_input=correct_answer,
                )
                _, rewards, _, _ = env.step(action)
                for k, v in rewards.items():
                    rewards_total[k] = rewards_total.get(k, 0.0) + float(v)

        elif phase.value == "post_discussion":
            # --- LLM-driven post-discussion conversations ---
            # Context & prompts come from env (canonical — same for all trainers)
            for sender in alive:
                for recipient in alive:
                    if sender == recipient:
                        continue
                    sender_veracity = agent_veracity_log.get(sender, "truth")

                    # Get structured context from the environment
                    disc_ctx = env.get_discussion_context(
                        agent_id=sender,
                        target_id=recipient,
                        agent_veracity=sender_veracity,
                    )

                    # Generate discussion message via LLM or fallback
                    content = ""
                    if use_llm_chat and policy_backend == "unsloth" and (unsloth_policies or unsloth_policy):
                        agent_policy = (unsloth_policies or {}).get(sender, unsloth_policy)
                        if agent_policy:
                            try:
                                # Use chat template (proper format for instruct models)
                                chat_msgs = PMEnvironment.build_discussion_messages(disc_ctx)
                                prompt = agent_policy.tokenizer.apply_chat_template(
                                    chat_msgs,
                                    tokenize=False,
                                    add_generation_prompt=True,
                                )
                                inputs = agent_policy.tokenizer(
                                    prompt, return_tensors="pt"
                                )
                                try:
                                    import torch
                                    if torch.cuda.is_available():
                                        inputs = {
                                            k: v.to("cuda")
                                            for k, v in inputs.items()
                                        }
                                except Exception:
                                    pass
                                out = agent_policy.model.generate(
                                    **inputs,
                                    max_new_tokens=50,
                                    do_sample=True,
                                    temperature=0.7,
                                    top_p=0.9,
                                    repetition_penalty=1.2,
                                )
                                # Decode only the new tokens
                                new_tokens = out[0][inputs["input_ids"].shape[-1]:]
                                content = agent_policy.tokenizer.decode(
                                    new_tokens, skip_special_tokens=True
                                ).strip()
                                # Strip quotes, trim to 2 sentences, cap length
                                content = content.strip('"').strip()
                                sentences = re.split(r'(?<=[.!?])\s+', content)
                                content = " ".join(sentences[:2])
                                if len(content) > 120:
                                    content = content[:117] + "..."
                            except Exception:
                                content = ""

                    if not content:
                        content = PMEnvironment.generate_discussion_fallback(disc_ctx)

                    chat_log.append({
                        "day": env.state.day,
                        "sender": sender,
                        "recipient": recipient,
                        "content": content,
                        "sender_veracity": sender_veracity,
                    })

                    msg = PostDiscussionMessage(
                        sender_id=sender,
                        recipient_id=recipient,
                        content=content,
                        day=env.state.day,
                        turn_index=0,
                    )
                    action = PMAction(
                        agent_id=sender,
                        action_type=ActionType.SEND_POST_DISCUSSION_MSG,
                        post_discussion_msg=msg,
                    )
                    try:
                        _, rewards, _, _ = env.step(action)
                        for k, v in rewards.items():
                            rewards_total[k] = rewards_total.get(k, 0.0) + float(v)
                    except ValueError:
                        pass

            # Trust assessments driven by observed veracity.
            # Check public reveal for exposed lies to inform trust decisions.
            reveal = env.state.public_reveals.get(env.state.day)
            for assessor in alive:
                for target in alive:
                    if assessor == target:
                        continue
                    # Decide trust delta based on what we know about target
                    if reveal and reveal.lies_unacknowledged.get(target, 0) > 0:
                        delta = TrustDelta.STRONG_DECREASE
                        reasoning = "caught lying"
                    elif reveal and reveal.lies_acknowledged.get(target, 0) > 0:
                        delta = TrustDelta.DECREASE
                        reasoning = "acknowledged lie"
                    else:
                        # Check pre-task messages for veracity
                        obs = env.get_observation(assessor)
                        target_msgs = [
                            m for m in obs.pre_task_messages_received
                            if m.sender_id == target
                        ]
                        if target_msgs and target_msgs[-1].veracity == MessageVeracity.TRUTH:
                            # Paranoia mechanic: highly deceptive agents discount positive trust signals
                            assessor_deception = env.agents[assessor].deception_prior
                            if assessor_deception > 0.4:
                                delta = TrustDelta.NEUTRAL
                                reasoning = "told truth but I am paranoid"
                            else:
                                delta = TrustDelta.INCREASE
                                reasoning = "told truth"
                        elif target_msgs and target_msgs[-1].veracity == MessageVeracity.LIE:
                            assessor_deception = env.agents[assessor].deception_prior
                            # Paranoia mechanic: deceptive agents amplify negative trust signals
                            if assessor_deception > 0.4:
                                delta = TrustDelta.STRONG_DECREASE
                                reasoning = "suspicious and I am paranoid"
                            else:
                                delta = TrustDelta.DECREASE
                                reasoning = "suspicious"
                        else:
                            delta = TrustDelta.NEUTRAL
                            reasoning = "no strong signal"
                    ta = TrustAssessment(
                        assessor_id=assessor,
                        target_id=target,
                        day=env.state.day,
                        reasoning=reasoning,
                        delta=delta,
                    )
                    action = PMAction(
                        agent_id=assessor,
                        action_type=ActionType.SUBMIT_TRUST_ASSESSMENT,
                        trust_assessment=ta,
                    )
                    _, rewards, _, _ = env.step(action)
                    for k, v in rewards.items():
                        rewards_total[k] = rewards_total.get(k, 0.0) + float(v)

        elif phase.value == "voting":
            for voter in alive:
                candidates = [x for x in alive if x != voter]
                # Sample vote strategy from learned policy
                vote_logits = policy_logits[voter].get("vote_strategy", {})
                if not vote_logits:
                    vote_logits = {"lowest_trust": 0.0, "highest_threat": 0.0, "random": 0.0}
                strategy = _sample_from(vote_logits)
                action_counts[voter][f"vote_{strategy}"] += 1

                trust_scores = env.state.trust_scores_dict.get(voter, {})
                if strategy == "lowest_trust":
                    # Vote for agent we trust least
                    target = min(candidates, key=lambda c: trust_scores.get(c, 0.5))
                elif strategy == "highest_threat":
                    # Vote for agent with highest points (biggest threat)
                    target = max(
                        candidates,
                        key=lambda c: env.state.agents_point_map.get(c, 0),
                    )
                else:
                    target = random.choice(candidates)

                action = PMAction(
                    agent_id=voter,
                    action_type=ActionType.VOTE,
                    vote_target=target,
                    task_input=f"{strategy} vote",
                )
                _, rewards, _, _ = env.step(action)
                for k, v in rewards.items():
                    rewards_total[k] = rewards_total.get(k, 0.0) + float(v)

        else:
            # TASK_REVEAL auto-advances in reset; this is a safety fallback.
            break

    # End-of-episode trust dispersion metric.
    trust_vals: List[float] = []
    for aid in env.state.alive_agents:
        trust_vals.extend(list(env.state.trust_scores_dict.get(aid, {}).values()))
    trust_dispersion = 0.0
    if trust_vals:
        mean = sum(trust_vals) / len(trust_vals)
        trust_dispersion = sum(abs(x - mean) for x in trust_vals) / len(trust_vals)

    trust_metrics = {"trust_dispersion": trust_dispersion}
    return rewards_total, action_counts, trust_metrics, chat_log


def update_policy(
    policy_logits: Dict[int, Dict],
    rewards: Dict[int, float],
    action_counts: Dict[int, Dict[str, int]],
    baseline: float,
    lr: float,
    entropy_coef: float = 0.02,
) -> None:
    # 1. Compute base advantages (reward - moving average)
    agent_advs = {aid: rewards.get(aid, 0.0) - baseline for aid in action_counts}
    
    # 2. Normalize advantages across agents in THIS episode.
    # This is crucial: it creates a strong relative signal (mean=0, std=1) 
    # forcing agents to differentiate instead of all collapsing to the entropy prior.
    adv_vals = list(agent_advs.values())
    if len(adv_vals) > 1:
        mean_a = sum(adv_vals) / len(adv_vals)
        std_a = (sum((x - mean_a) ** 2 for x in adv_vals) / len(adv_vals)) ** 0.5
        if std_a > 1e-5:
            agent_advs = {aid: (v - mean_a) / std_a for aid, v in agent_advs.items()}
        else:
            agent_advs = {aid: v - mean_a for aid, v in agent_advs.items()}

    for aid, counts in action_counts.items():
        advantage = agent_advs.get(aid, 0.0)

        # --- Update veracity policy ---
        veracity_labels = ("truth", "twist", "lie")
        v_total = sum(counts.get(l, 0) for l in veracity_labels)
        if v_total > 0:
            v_probs = _softmax({l: policy_logits[aid][l] for l in veracity_labels})
            for label in veracity_labels:
                frac = counts[label] / v_total
                policy_logits[aid][label] += lr * advantage * frac
                policy_logits[aid][label] += entropy_coef * (1.0 / 3.0 - v_probs[label])

        # --- Update vote strategy policy ---
        vote_labels = ("lowest_trust", "highest_threat", "random")
        vote_logits = policy_logits[aid].setdefault(
            "vote_strategy", {l: 0.0 for l in vote_labels}
        )
        vote_total = sum(counts.get(f"vote_{l}", 0) for l in vote_labels)
        if vote_total > 0:
            vote_probs = _softmax(vote_logits)
            for label in vote_labels:
                frac = counts.get(f"vote_{label}", 0) / vote_total
                vote_logits[label] += lr * advantage * frac
                vote_logits[label] += entropy_coef * (1.0 / 3.0 - vote_probs[label])


def main() -> None:
    parser = argparse.ArgumentParser(description="Online RL trainer for Project Machiavelli.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--entropy_coef", type=float, default=0.02)
    parser.add_argument("--llm_chat_every", type=int, default=20, 
                        help="Only generate LLM text every N episodes to speed up training")
    parser.add_argument(
        "--policy_backend",
        choices=["tabular", "unsloth", "hf_endpoint"],
        default="tabular",
        help="tabular = fast REINFORCE logits; unsloth/hf_endpoint = LM-driven action generation",
    )
    parser.add_argument(
        "--unsloth_model",
        default="unsloth/Qwen2.5-0.5B-Instruct",
        help="Model used when --policy_backend unsloth",
    )
    parser.add_argument(
        "--unsloth_per_agent_models",
        nargs="+",
        default=None,
        help=(
            "Optional per-agent model list for independent entities. "
            "Example for 3 agents: --unsloth_per_agent_models modelA modelB modelC"
        ),
    )
    parser.add_argument(
        "--hf_model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF model used when --policy_backend hf_endpoint",
    )
    parser.add_argument(
        "--hf_per_agent_models",
        nargs="+",
        default=None,
        help=(
            "Optional per-agent HF model list for independent entities. "
            "Example: --hf_per_agent_models modelA modelB modelC"
        ),
    )
    parser.add_argument(
        "--hf_token_env",
        default="HF_TOKEN",
        help="Environment variable name for HF token (used by hf_endpoint backend).",
    )
    parser.add_argument("--out_csv", default="logs/online_rl_metrics.csv")
    parser.add_argument("--out_json", default="logs/online_rl_summary.json")
    parser.add_argument("--out_plot", default="logs/online_rl_plots.png")
    args = parser.parse_args()

    random.seed(args.seed)
    env = PMEnvironment()

    # Initialize once with reset to know agent ids, then policy carries across episodes.
    env.reset(task=args.difficulty)
    agent_ids = list(env.agents.keys())
    policy_logits = {
        aid: {
            "truth": 0.0, "twist": 0.0, "lie": 0.0,
            "vote_strategy": {
                "lowest_trust": 0.0, "highest_threat": 0.0, "random": 0.0,
            },
        }
        for aid in agent_ids
    }
    unsloth_policy: Optional[UnslothPolicy] = None
    unsloth_policies: Optional[Dict[int, UnslothPolicy]] = None
    hf_policy: Optional[HFEndpointPolicy] = None
    hf_policies: Optional[Dict[int, HFEndpointPolicy]] = None
    if args.policy_backend == "unsloth":
        if args.unsloth_per_agent_models:
            if len(args.unsloth_per_agent_models) != len(agent_ids):
                raise ValueError(
                    f"--unsloth_per_agent_models expects {len(agent_ids)} models, "
                    f"got {len(args.unsloth_per_agent_models)}."
                )
            print("Loading independent Unsloth model per agent...")
            unsloth_policies = {}
            for aid, model_name in zip(agent_ids, args.unsloth_per_agent_models):
                print(f"  Agent {aid} -> {model_name}")
                unsloth_policies[aid] = UnslothPolicy(model_name=model_name)
            # Fallback handle, should not be used when unsloth_policies is present.
            unsloth_policy = next(iter(unsloth_policies.values()))
        else:
            print(f"Loading shared Unsloth model: {args.unsloth_model}")
            unsloth_policy = UnslothPolicy(model_name=args.unsloth_model)
    elif args.policy_backend == "hf_endpoint":
        hf_token = os.environ.get(args.hf_token_env)
        if args.hf_per_agent_models:
            if len(args.hf_per_agent_models) != len(agent_ids):
                raise ValueError(
                    f"--hf_per_agent_models expects {len(agent_ids)} models, "
                    f"got {len(args.hf_per_agent_models)}."
                )
            print("Loading independent HF endpoint model per agent...")
            hf_policies = {}
            for aid, model_name in zip(agent_ids, args.hf_per_agent_models):
                print(f"  Agent {aid} -> {model_name}")
                hf_policies[aid] = HFEndpointPolicy(model_name=model_name, token=hf_token)
            hf_policy = next(iter(hf_policies.values()))
        else:
            print(f"Loading shared HF endpoint model: {args.hf_model}")
            hf_policy = HFEndpointPolicy(model_name=args.hf_model, token=hf_token)

    rows = []
    reward_history: List[float] = []
    chat_dir = Path("logs/chats")
    chat_dir.mkdir(parents=True, exist_ok=True)
    
    tb_writer = SummaryWriter(log_dir="logs/tensorboard")

    for ep in range(1, args.episodes + 1):
        # Only use slow LLM generation every N episodes to speed up training
        use_llm_chat = (ep % args.llm_chat_every == 0) or (ep == args.episodes)

        rewards, action_counts, trust_metrics, chat_log = run_episode(
            env=env,
            policy_logits=policy_logits,
            difficulty=args.difficulty,
            policy_backend=args.policy_backend,
            unsloth_policy=unsloth_policy,
            unsloth_policies=unsloth_policies,
            hf_policy=hf_policy,
            hf_policies=hf_policies,
            use_llm_chat=use_llm_chat,
        )

        mean_reward = sum(rewards.values()) / len(rewards) if rewards else 0.0
        reward_history.append(mean_reward)
        baseline = sum(reward_history) / len(reward_history)
        update_policy(
            policy_logits=policy_logits,
            rewards=rewards,
            action_counts=action_counts,
            baseline=baseline,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
        )

        # --- Sync policy_logits → agent priors (so they carry across episodes) ---
        for aid in agent_ids:
            v_probs = _softmax({
                "truth": policy_logits[aid]["truth"],
                "twist": policy_logits[aid]["twist"],
                "lie":   policy_logits[aid]["lie"],
            })
            new_truthful  = round(v_probs["truth"], 4)
            new_deception = round(v_probs["lie"], 4)
            # risk_beta: higher when agent lies more (risk-seeking), lower when truthful
            new_risk_beta = round(max(0.1, 0.5 + 2.0 * (v_probs["lie"] - v_probs["truth"])), 4)

            if aid in env.agents:
                env.agents[aid].truthful_prior  = new_truthful
                env.agents[aid].deception_prior = new_deception
                env.agents[aid].risk_beta       = new_risk_beta

            # Also update global_inference_store so next env.reset() uses these
            env.global_inference_store.agent_priors[aid] = AgentPriorSnapshot(
                agent_id=aid,
                episode_index=ep,
                truthful_prior=new_truthful,
                deception_prior=new_deception,
                risk_beta=new_risk_beta,
                final_trust_scores=dict(
                    env.state.trust_scores_dict.get(aid, {})
                ),
            )

        lie_count = sum(v["lie"] for v in action_counts.values())
        truth_count = sum(v["truth"] for v in action_counts.values())
        twist_count = sum(v["twist"] for v in action_counts.values())
        total_count = max(1, lie_count + truth_count + twist_count)

        vlt = sum(v.get("vote_lowest_trust", 0) for v in action_counts.values())
        vht = sum(v.get("vote_highest_threat", 0) for v in action_counts.values())
        vrn = sum(v.get("vote_random", 0) for v in action_counts.values())
        vote_total = max(1, vlt + vht + vrn)

        # Collect per-agent priors for logging
        avg_truthful  = sum(
            _softmax({k: policy_logits[a][k] for k in ("truth","twist","lie")})["truth"]
            for a in agent_ids
        ) / len(agent_ids)
        avg_deception = sum(
            _softmax({k: policy_logits[a][k] for k in ("truth","twist","lie")})["lie"]
            for a in agent_ids
        ) / len(agent_ids)
        avg_risk_beta = sum(
            max(0.1, 0.5 + 2.0 * (
                _softmax({k: policy_logits[a][k] for k in ("truth","twist","lie")})["lie"]
                - _softmax({k: policy_logits[a][k] for k in ("truth","twist","lie")})["truth"]
            ))
            for a in agent_ids
        ) / len(agent_ids)

        row = {
            "episode": ep,
            "mean_reward": round(mean_reward, 4),
            "baseline_reward": round(baseline, 4),
            "lie_rate": round(lie_count / total_count, 4),
            "truth_rate": round(truth_count / total_count, 4),
            "twist_rate": round(twist_count / total_count, 4),
            "vote_lowest_trust": round(vlt / vote_total, 4),
            "vote_highest_threat": round(vht / vote_total, 4),
            "vote_random": round(vrn / vote_total, 4),
            "trust_dispersion": round(trust_metrics["trust_dispersion"], 4),
            "avg_truthful_prior": round(avg_truthful, 4),
            "avg_deception_prior": round(avg_deception, 4),
            "avg_risk_beta": round(avg_risk_beta, 4),
        }
        # Per-agent detail: priors + individual rewards
        for a in agent_ids:
            vp = _softmax({k: policy_logits[a][k] for k in ("truth","twist","lie")})
            row[f"agent{a}_truthful"]  = round(vp["truth"], 4)
            row[f"agent{a}_deception"] = round(vp["lie"], 4)
            row[f"agent{a}_risk_beta"] = round(max(0.1, 0.5 + 2.0 * (vp["lie"] - vp["truth"])), 4)
            row[f"agent{a}_reward"]    = round(rewards.get(a, 0.0), 4)

        rows.append(row)

        # Build per-agent reward string
        agent_rewards_str = " ".join(
            f"A{a}:{rewards.get(a, 0.0):.2f}" for a in agent_ids
        )
        print(
            f"[ep {ep:03d}] rewards=[{agent_rewards_str}] mean={row['mean_reward']:.3f} "
            f"lie={row['lie_rate']:.2f} truth={row['truth_rate']:.2f} "
            f"twist={row['twist_rate']:.2f} "
            f"vote=LT{row['vote_lowest_trust']:.0%}/HT{row['vote_highest_threat']:.0%}/R{row['vote_random']:.0%} "
            f"trust_disp={row['trust_dispersion']:.3f} "
            f"priors=[T:{row['avg_truthful_prior']:.2f} D:{row['avg_deception_prior']:.2f} β:{row['avg_risk_beta']:.2f}]"
        )

        # --- Tensorboard Logging ---
        tb_writer.add_scalar("Globals/Mean_Reward", row["mean_reward"], ep)
        tb_writer.add_scalar("Globals/Trust_Dispersion", row["trust_dispersion"], ep)
        
        # Plot globals behavior mix
        tb_writer.add_scalar("Globals_Behavior/Lie_Rate", row["lie_rate"], ep)
        tb_writer.add_scalar("Globals_Behavior/Truth_Rate", row["truth_rate"], ep)
        tb_writer.add_scalar("Globals_Behavior/Twist_Rate", row["twist_rate"], ep)

        # Plot per-agent metrics
        truth_priors = {}
        deception_priors = {}
        risk_betas = {}
        rewards_dict = {}
        behav_truth = {}
        behav_lie = {}
        behav_twist = {}

        for a in agent_ids:
            truth_priors[f"Agent_{a}"] = row[f"agent{a}_truthful"]
            deception_priors[f"Agent_{a}"] = row[f"agent{a}_deception"]
            risk_betas[f"Agent_{a}"] = row[f"agent{a}_risk_beta"]
            rewards_dict[f"Agent_{a}"] = row[f"agent{a}_reward"]

            # Behaviour mix (action rates per agent)
            total_acts_a = max(1, action_counts[a].get("truth", 0) + action_counts[a].get("lie", 0) + action_counts[a].get("twist", 0))
            behav_truth[f"Agent_{a}"] = action_counts[a].get("truth", 0) / total_acts_a
            behav_lie[f"Agent_{a}"] = action_counts[a].get("lie", 0) / total_acts_a
            behav_twist[f"Agent_{a}"] = action_counts[a].get("twist", 0) / total_acts_a

        tb_writer.add_scalars("Priors/Truthful", truth_priors, ep)
        tb_writer.add_scalars("Priors/Deception", deception_priors, ep)
        tb_writer.add_scalars("Priors/Risk_Beta", risk_betas, ep)
        tb_writer.add_scalars("Rewards/Per_Agent", rewards_dict, ep)
        tb_writer.add_scalars("Behavior_Mix/Truth", behav_truth, ep)
        tb_writer.add_scalars("Behavior_Mix/Lie", behav_lie, ep)
        tb_writer.add_scalars("Behavior_Mix/Twist", behav_twist, ep)
        
        # Flush TensorBoard writer so data is visible immediately and isn't lost on Ctrl+C
        tb_writer.flush()

        # --- Display post-discussion chats in terminal ---
        if chat_log:
            days_in_ep = sorted(set(m["day"] for m in chat_log))
            for d in days_in_ep:
                _display_chat(chat_log, d)

        # --- Save HTML chat log ---
        if chat_log:
            html_path = chat_dir / f"ep_{ep:03d}.html"
            _save_chat_html(chat_log, ep, html_path)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    first_n = rows[: max(1, len(rows) // 3)]
    last_n = rows[-max(1, len(rows) // 3) :]

    def avg(key: str, items: List[Dict]) -> float:
        return sum(float(i[key]) for i in items) / len(items)

    summary = {
        "episodes": args.episodes,
        "difficulty": args.difficulty,
        "avg_reward_first_third": round(avg("mean_reward", first_n), 4),
        "avg_reward_last_third": round(avg("mean_reward", last_n), 4),
        "avg_lie_rate_first_third": round(avg("lie_rate", first_n), 4),
        "avg_lie_rate_last_third": round(avg("lie_rate", last_n), 4),
        "avg_trust_dispersion_first_third": round(avg("trust_dispersion", first_n), 4),
        "avg_trust_dispersion_last_third": round(avg("trust_dispersion", last_n), 4),
        "policy_backend": args.policy_backend,
        "entity_mode": (
            "independent_unsloth_per_agent"
            if args.policy_backend == "unsloth" and bool(args.unsloth_per_agent_models)
            else "independent_hf_per_agent"
            if args.policy_backend == "hf_endpoint" and bool(args.hf_per_agent_models)
            else "shared_policy"
        ),
        "unsloth_model": args.unsloth_model if args.policy_backend == "unsloth" else None,
        "unsloth_per_agent_models": args.unsloth_per_agent_models,
        "hf_model": args.hf_model if args.policy_backend == "hf_endpoint" else None,
        "hf_per_agent_models": args.hf_per_agent_models,
        "entropy_coef": args.entropy_coef,
        "final_policy_logits": policy_logits,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ── Visualizations ─────────────────────────────────────────────────────
    episodes = [r["episode"] for r in rows]
    rewards = [r["mean_reward"] for r in rows]
    baselines = [r["baseline_reward"] for r in rows]
    lie_rates = [r["lie_rate"] for r in rows]
    truth_rates = [r["truth_rate"] for r in rows]
    twist_rates = [r["twist_rate"] for r in rows]
    trust_disp = [r["trust_dispersion"] for r in rows]

    # 5-episode moving average for smoother trend view.
    win = 5
    reward_ma = []
    for i in range(len(rewards)):
        left = max(0, i - win + 1)
        chunk = rewards[left : i + 1]
        reward_ma.append(sum(chunk) / len(chunk))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (1) Reward trajectory
    ax = axes[0, 0]
    ax.plot(episodes, rewards, label="mean_reward", linewidth=1.8)
    ax.plot(episodes, baselines, label="baseline_reward", linestyle="--", linewidth=1.2)
    ax.plot(episodes, reward_ma, label="reward_ma_5", linestyle=":", linewidth=1.8)
    ax.set_title("Reward Trends")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25)
    ax.legend()

    # (2) Veracity behavior mix
    ax = axes[0, 1]
    ax.plot(episodes, lie_rates, label="lie_rate", linewidth=1.8)
    ax.plot(episodes, truth_rates, label="truth_rate", linewidth=1.8)
    ax.plot(episodes, twist_rates, label="twist_rate", linewidth=1.8)
    ax.set_title("Behavior Mix")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()

    # (3) Trust dispersion trend
    ax = axes[1, 0]
    ax.plot(episodes, trust_disp, label="trust_dispersion", color="purple", linewidth=1.8)
    ax.set_title("Trust Dispersion")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Dispersion")
    ax.grid(alpha=0.25)
    ax.legend()

    # (4) Early vs late aggregate comparison
    ax = axes[1, 1]
    metrics = ["reward", "lie_rate", "trust_disp"]
    early_vals = [
        summary["avg_reward_first_third"],
        summary["avg_lie_rate_first_third"],
        summary["avg_trust_dispersion_first_third"],
    ]
    late_vals = [
        summary["avg_reward_last_third"],
        summary["avg_lie_rate_last_third"],
        summary["avg_trust_dispersion_last_third"],
    ]
    x = list(range(len(metrics)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], early_vals, width=width, label="first_third")
    ax.bar([i + width / 2 for i in x], late_vals, width=width, label="last_third")
    ax.set_title("Early vs Late")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.suptitle(
        f"Online RL Diagnostics ({args.policy_backend}) | difficulty={args.difficulty}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_plot = Path(args.out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)

    print(f"\nWrote episode metrics: {out_csv}")
    print(f"Wrote summary: {out_json}")
    print(f"Wrote plots: {out_plot}")
    print(f"Wrote chat logs: {chat_dir}/")
    print(f"Wrote TensorBoard logs: logs/tensorboard/")
    print(json.dumps(summary, indent=2))
    
    tb_writer.close()


if __name__ == "__main__":
    main()
