from graders.easy import EasyGrader
from graders.medium import MediumGrader
from graders.hard import HardGrader


_GRADER_MAP = {
    "easy":   EasyGrader,
    "medium": MediumGrader,
    "hard":   HardGrader,
}


def get_grader(difficulty: str):
    """
    Return the correct grader instance for the given difficulty.

    Parameters
    ----------
    difficulty : "easy" | "medium" | "hard"
    """
    difficulty = difficulty.replace("task_", "")
    if difficulty not in _GRADER_MAP:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Must be one of: {list(_GRADER_MAP.keys())}"
        )
    return _GRADER_MAP[difficulty]()


__all__ = ['EasyGrader', 'MediumGrader', 'HardGrader', 'get_grader']