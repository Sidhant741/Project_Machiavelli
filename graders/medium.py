"""
medium.py — Project Machiavelli Grader
========================================
Medium difficulty grader.

Config:
    n_options               = 2   (50/50 without help from others)
    lie_penalty             = 2   (point deduction for lying)
    lie_acknowledged_penalty= 1   (point deduction for TWIST)

All weights controlled via grader_config.py GRADER_CONFIG["medium"].
All reward components imported from helpers.py.
"""

from __future__ import annotations

from typing import Dict

try:
    from ..models import PMState
    from .grader_config import GRADER_CONFIG
    from . import helpers
except ImportError:
    from models import PMState
    from grader_config import GRADER_CONFIG
    import helpers


CFG = GRADER_CONFIG["medium"]


def grade(agent_id: int, state: PMState, task_config: Dict) -> float:
    r  = helpers.task_score(agent_id, state, task_config, CFG)
    r += helpers.influence_score(agent_id, state, CFG)
    r += helpers.jury_win(agent_id, state, CFG)
    r += helpers.lie_caught(agent_id, state, CFG)
    r += helpers.lie_exposed(agent_id, state, CFG)
    r += helpers.survival_streak(agent_id, state, CFG)
    return helpers.normalise(r, CFG)

class MediumGrader:
    """Callable grader for medium difficulty."""

    def __call__(
        self,
        agent_id: int,
        state: PMState,
        task_config: Dict,
    ) -> float:
        return grade(agent_id, state, task_config)

    def grade_all(self, state: PMState, task_config: Dict) -> Dict[int, float]:
        """Compute rewards for every alive agent at once."""
        return {aid: grade(aid, state, task_config) for aid in state.alive_agents}