"""
grader_config.py — Project Machiavelli
========================================
Central config for all grader reward weights.

Modify the values here to tune agent incentives per difficulty.
All weights are applied in easy.py / medium.py / hard.py.

Component guide:
    task_score      — correct answer points / max points  * w
    influence_score — voted for the eliminated agent      * w
    jury_win        — won the final jury vote             * w  (flat bonus)
    lie_caught      — voted for someone the reveal confirmed lied  * w
    lie_exposed     — this agent lied and got exposed     * w  (applied as negative)
    survival_streak — consecutive days with 0 votes received * w (per day in streak)
"""

GRADER_CONFIG = {
    "easy": {
        "task_score":       0.3,
        "influence_score":  0.5,
        "jury_win":         2.0,
        "lie_caught":       0.8,
        "lie_exposed":      0.0,
        "survival_streak":  0.0,
        "max_reward":       3.6,   # 0.3 + 0.5 + 0.8 + 2.0
    },
    "medium": {
        "task_score":       0.3,
        "influence_score":  0.7,
        "jury_win":         2.0,
        "lie_caught":       0.8,
        "lie_exposed":      0.7,
        "survival_streak":  0.0,
        "max_reward":       3.8,   # 0.3 + 0.7 + 0.8 + 2.0
    },
    "hard": {
        "task_score":       0.3,
        "influence_score":  1.0,
        "jury_win":         2.0,
        "lie_caught":       0.8,
        "lie_exposed":      0.7,
        "survival_streak":  0.2,
        "max_reward":       4.7,   # 0.3 + 1.0 + 0.8 + 0.6 + 2.0
    },
}