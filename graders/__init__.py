"""
graders/__init__.py — Project Machiavelli
==========================================
Dispatcher: get_grader(difficulty) returns the correct grader instance.

Usage
-----
from graders import get_grader

grader = get_grader("easy")
rewards = grader.grade_all(state, task_config)
# → { agent_id: float }

# Or per-agent:
reward = grader(agent_id, state, task_config)
"""

from __future__ import annotations

from typing import Union

try:
    from .easy   import EasyGrader
    from .medium import MediumGrader
    from .hard   import HardGrader
except ImportError:
    from graders.easy   import EasyGrader
    from graders.medium import MediumGrader
    from graders.hard   import HardGrader


# Type alias for grader instances
AnyGrader = Union[EasyGrader, MediumGrader, HardGrader]

_GRADER_MAP = {
    "easy":   EasyGrader,
    "medium": MediumGrader,
    "hard":   HardGrader,
}


def get_grader(difficulty: str) -> AnyGrader:
    """
    Return the correct grader instance for the given difficulty.

    Parameters
    ----------
    difficulty : "easy" | "medium" | "hard"

    Returns
    -------
    EasyGrader | MediumGrader | HardGrader
    """
    difficulty = difficulty.replace("task_", "")
    if difficulty not in _GRADER_MAP:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Must be one of: {list(_GRADER_MAP.keys())}"
        )
    return _GRADER_MAP[difficulty]()


__all__ = [
    "get_grader",
    "EasyGrader",
    "MediumGrader",
    "HardGrader",
]