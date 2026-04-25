"""
helpers.py — Project Machiavelli Grader
=========================================
Shared reward component functions used by easy.py, medium.py, hard.py.

Each function takes a cfg dict (the difficulty's GRADER_CONFIG entry)
so the weight comes from the caller, not hardcoded here.

Components:
    task_score       — correct answer fraction * weight
    influence_score  — voted for the eliminated agent * weight
    jury_win         — won the jury vote * weight (flat bonus)
    lie_caught       — voted for a confirmed liar * weight
    lie_exposed      — this agent's lie was publicly exposed * weight (negative)
    survival_streak  — consecutive days with 0 votes received * weight
"""

from __future__ import annotations

from typing import Dict

try:
    from ..models import PMState
except ImportError:
    from models import PMState


def task_score(agent_id: int, state: PMState, task_config: Dict, cfg: Dict) -> float:
    result = state.task_results.get(state.day)
    if result is None:
        return 0.0
    day_score = result.per_agent_outcome.get(agent_id, 0)
    max_pts   = task_config["correct_answer_points"]
    return (day_score / max_pts) * cfg["task_score"]


def influence_score(agent_id: int, state: PMState, cfg: Dict) -> float:
    if not state.vote_history:
        return 0.0
    last_vote  = state.vote_history[-1]
    my_vote    = last_vote.votes_cast.get(agent_id)
    eliminated = last_vote.eliminated_id
    if my_vote is None or eliminated is None:
        return 0.0
    return cfg["influence_score"] if my_vote == eliminated else 0.0


def jury_win(agent_id: int, state: PMState, cfg: Dict) -> float:
    return cfg["jury_win"] if state.game_winner == agent_id else 0.0


def lie_caught(agent_id: int, state: PMState, cfg: Dict) -> float:
    reveal = state.public_reveals.get(state.day)
    if not state.vote_history or reveal is None:
        return 0.0
    my_vote = state.vote_history[-1].votes_cast.get(agent_id)
    if my_vote is None:
        return 0.0
    return cfg["lie_caught"] if reveal.lies_unacknowledged.get(my_vote, 0) > 0 else 0.0


def lie_exposed(agent_id: int, state: PMState, cfg: Dict) -> float:
    reveal = state.public_reveals.get(state.day)
    if reveal is None:
        return 0.0
    return -cfg["lie_exposed"] if reveal.lies_unacknowledged.get(agent_id, 0) > 0 else 0.0


def survival_streak(agent_id: int, state: PMState, cfg: Dict) -> float:
    """
    Count consecutive days (going backwards from today) where this agent
    received zero votes. Multiply by survival_streak weight.

    Example:
        Day 1: votes_received = 0  ─┐
        Day 2: votes_received = 0   ├─ streak = 3  →  3 * 0.2 = 0.6
        Day 3: votes_received = 0  ─┘
        Day 4: votes_received = 2  → streak broken, stop counting
    """
    streak = 0
    for record in reversed(state.vote_history):
        if record.vote_counts.get(agent_id, 0) == 0:
            streak += 1
        else:
            break
    return streak * cfg["survival_streak"]

def normalise(raw: float, cfg: Dict) -> float:
    """Clip raw reward to [0, 1] using max_reward from config."""
    normalised = raw / cfg["max_reward"]
    return round(float(np.clip(normalised, 0.0, 1.0)), 4)