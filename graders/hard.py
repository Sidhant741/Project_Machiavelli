"""
hard.py — Project Machiavelli Grader
======================================
Hard difficulty grader.

Config:
    n_options               = 3   (1-in-3 without help from others)
    lie_penalty             = 5   (heavy point deduction for lying)
    lie_acknowledged_penalty= 2   (point deduction for TWIST)

All weights controlled via grader_config.py GRADER_CONFIG["hard"].
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
    from graders.grader_config import GRADER_CONFIG
    import graders.helper as helpers


CFG = GRADER_CONFIG["hard"]


def grade(agent_id: int, state: PMState, task_config: Dict) -> float:
    r  = helpers.task_score(agent_id, state, task_config, CFG)
    r += helpers.influence_score(agent_id, state, CFG)
    r += helpers.jury_win(agent_id, state, CFG)
    r += helpers.lie_caught(agent_id, state, CFG)
    r += helpers.lie_exposed(agent_id, state, CFG)
    r += helpers.deception_success(agent_id, state, CFG)
    r += helpers.strategic_deception(agent_id, state, CFG)
    r += helpers.survival_streak(agent_id, state, CFG)
    return helpers.normalise(r, CFG)

class HardGrader:
    """Callable grader for hard difficulty."""

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