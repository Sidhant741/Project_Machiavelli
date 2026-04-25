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
    normalise        — clip raw reward to [0, 1] using max_reward from config
"""

from __future__ import annotations

from typing import Dict

try:
    from ..models import PMState
except ImportError:
    from models import PMState


# ---------------------------------------------------------------------------
# Task score
# ---------------------------------------------------------------------------

def task_score(
    agent_id: int,
    state: PMState,
    task_config: Dict,
    cfg: Dict,
) -> float:
    """
    Fraction of max possible task points this agent earned today, * weight.
    Returns 0.0 if no task result exists yet.
    """
    result = state.task_results.get(state.day)
    if result is None:
        return 0.0
    day_score = result.per_agent_outcome.get(agent_id, 0)
    max_pts   = task_config.get("correct_answer_points", 1)
    if max_pts == 0:
        return 0.0
    return (day_score / max_pts) * cfg["task_score"]


# ---------------------------------------------------------------------------
# Influence score
# ---------------------------------------------------------------------------

def influence_score(agent_id: int, state: PMState, cfg: Dict) -> float:
    """
    +weight if this agent voted for the agent who was actually eliminated.
    Rewards agents who read the room correctly and voted with the majority.
    """
    if not state.vote_history:
        return 0.0
    last_vote  = state.vote_history[-1]
    my_vote    = last_vote.votes_cast.get(agent_id)
    eliminated = last_vote.eliminated_id
    if my_vote is None or eliminated is None:
        return 0.0
    return cfg["influence_score"] if my_vote == eliminated else 0.0


# ---------------------------------------------------------------------------
# Jury win
# ---------------------------------------------------------------------------

def jury_win(agent_id: int, state: PMState, cfg: Dict) -> float:
    """
    Flat bonus awarded to the game winner (last agent standing).
    game_winner is set on PMState when only one agent remains.
    Returns 0.0 during normal days — only non-zero on the final day.
    """
    winner = getattr(state, "game_winner", None)
    if winner is None:
        return 0.0
    return cfg["jury_win"] if winner == agent_id else 0.0


# ---------------------------------------------------------------------------
# Lie caught
# ---------------------------------------------------------------------------

def lie_caught(agent_id: int, state: PMState, cfg: Dict) -> float:
    """
    +weight if this agent voted for someone whose unacknowledged lie
    was publicly confirmed in today's reveal.
    Rewards agents who correctly identified and punished deceivers.
    """
    if not state.vote_history:
        return 0.0
    reveal = state.public_reveals.get(state.day)
    if reveal is None:
        return 0.0
    my_vote = state.vote_history[-1].votes_cast.get(agent_id)
    if my_vote is None:
        return 0.0
    return cfg["lie_caught"] if reveal.lies_unacknowledged.get(my_vote, 0) > 0 else 0.0


# ---------------------------------------------------------------------------
# Lie exposed
# ---------------------------------------------------------------------------

def lie_exposed(agent_id: int, state: PMState, cfg: Dict) -> float:
    """
    -weight if this agent's own unacknowledged lie was publicly exposed today.
    Applied as a negative: penalises agents who lied and didn't confess.
    Returns 0.0 if cfg["lie_exposed"] == 0 (easy difficulty).
    """
    if cfg.get("lie_exposed", 0.0) == 0.0:
        return 0.0
    reveal = state.public_reveals.get(state.day)
    if reveal is None:
        return 0.0
    return -cfg["lie_exposed"] if reveal.lies_unacknowledged.get(agent_id, 0) > 0 else 0.0


# ---------------------------------------------------------------------------
# Survival streak
# ---------------------------------------------------------------------------

def survival_streak(agent_id: int, state: PMState, cfg: Dict) -> float:
    """
    Count consecutive days (going backwards from today) where this agent
    received zero votes. Multiply count by survival_streak weight.

    Example (hard difficulty, weight=0.2):
        Day 1: votes_received = 0  ─┐
        Day 2: votes_received = 0   ├── streak = 3 → 3 * 0.2 = 0.6
        Day 3: votes_received = 0  ─┘
        Day 4: votes_received = 2  → streak broken, stop
    Returns 0.0 if cfg["survival_streak"] == 0 (easy / medium).
    """
    if cfg.get("survival_streak", 0.0) == 0.0:
        return 0.0
    streak = 0
    for record in reversed(state.vote_history):
        if record.vote_counts.get(agent_id, 0) == 0:
            streak += 1
        else:
            break
    return streak * cfg["survival_streak"]


# ---------------------------------------------------------------------------
# Normalise
# ---------------------------------------------------------------------------

def normalise(raw: float, cfg: Dict) -> float:
    """
    Clip raw reward to [0.0, 1.0] using cfg["max_reward"] as the ceiling.
    Returns a rounded float.
    """
    max_r = cfg.get("max_reward", 1.0)
    if max_r == 0:
        return 0.0
    normalised = raw / max_r
    # Manual clip — avoid numpy dependency
    normalised = max(0.0, min(1.0, normalised))
    return round(float(normalised), 4)