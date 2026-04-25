"""
easy.py — Project Machiavelli Grader
======================================
Easy difficulty grader.

Config:
    n_options   = 1   (agent gets correct answer directly)
    lie_penalty = 0   (no point deduction for lying)

All weights controlled via grader_config.py GRADER_CONFIG["easy"].
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


CFG = GRADER_CONFIG["easy"]


def grade(agent_id: int, state: PMState, task_config: Dict) -> float:
    r  = helpers.task_score(agent_id, state, task_config, CFG)
    r += helpers.influence_score(agent_id, state, CFG)
    r += helpers.jury_win(agent_id, state, CFG)
    r += helpers.lie_caught(agent_id, state, CFG)
    r += helpers.lie_exposed(agent_id, state, CFG)
    r += helpers.survival_streak(agent_id, state, CFG)
    return helpers.normalise(r, CFG)


class EasyGrader:
    """Callable grader for easy difficulty."""

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