"""
hard.py — Project Machiavelli Grader
======================================
Hard difficulty grader.

Config:
    n_options               = 3   (1-in-3 without help from others)
    lie_penalty             = 5   (heavy point deduction for lying)
    lie_acknowledged_penalty= 2   (point deduction for TWIST)

All weights controlled via grader_config.py GRADER_CONFIG["hard"].
All reward components imported from helper.py.
"""

from __future__ import annotations

from typing import Dict

try:
    from ..models import PMState
    from .grader_config import GRADER_CONFIG
    from . import helper
except ImportError:
    from models import PMState
    from graders.grader_config import GRADER_CONFIG
    import graders.helper as helper


CFG = GRADER_CONFIG["hard"]


def grade(agent_id: int, state: PMState, task_config: Dict) -> float:
    r  = helper.task_score(agent_id, state, task_config, CFG)
    r += helper.influence_score(agent_id, state, CFG)
    r += helper.jury_win(agent_id, state, CFG)
    r += helper.lie_caught(agent_id, state, CFG)
    r += helper.lie_exposed(agent_id, state, CFG)
    r += helper.deception_success(agent_id, state, CFG)
    r += helper.strategic_deception(agent_id, state, CFG)
    r += helper.survival_streak(agent_id, state, CFG)
    return helper.normalise(r, CFG)

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