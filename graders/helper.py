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
    deception_success— lied and survived voting (strategic success) * weight
    strategic_deception — lied + survived + voted correctly + low suspicion * weight
    survival_streak  — consecutive days with 0 votes received * weight
"""

from __future__ import annotations

from typing import Dict

try:
    from ..models import PMState
except ImportError:
    from models import PMState

import numpy as np


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


def deception_success(agent_id: int, state: PMState, cfg: Dict) -> float:
    """
    Reward successful deception:
    - agent told an unacknowledged lie today, and
    - agent was NOT eliminated in today's vote.
    """
    reveal = state.public_reveals.get(state.day)
    if reveal is None or not state.vote_history:
        return 0.0
    lied_unack = reveal.lies_unacknowledged.get(agent_id, 0) > 0
    if not lied_unack:
        return 0.0
    eliminated = state.vote_history[-1].eliminated_id
    survived = (eliminated != agent_id)
    return cfg.get("deception_success", 0.0) if survived else 0.0


def strategic_deception(agent_id: int, state: PMState, cfg: Dict) -> float:
    """
    Reward deception only when it appears strategically effective:
    - agent told an unacknowledged lie today
    - agent survived today's vote
    - agent voted for the eliminated target (influence alignment)
    - agent did not attract many votes (low suspicion proxy)
    """
    reveal = state.public_reveals.get(state.day)
    if reveal is None or not state.vote_history:
        return 0.0

    last_vote = state.vote_history[-1]
    eliminated = last_vote.eliminated_id
    if eliminated is None:
        return 0.0

    lied_unack = reveal.lies_unacknowledged.get(agent_id, 0) > 0
    if not lied_unack:
        return 0.0

    survived = (eliminated != agent_id)
    voted_with_majority = (last_vote.votes_cast.get(agent_id) == eliminated)
    low_suspicion = (last_vote.vote_counts.get(agent_id, 0) <= 1)

    if survived and voted_with_majority and low_suspicion:
        return cfg.get("strategic_deception", 0.0)
    return 0.0


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