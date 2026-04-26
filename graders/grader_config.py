"""
grader_config.py — Project Machiavelli
========================================
Central config for all grader reward weights.

Modify the values here to tune agent incentives per difficulty.
All weights are applied in easy.py / medium.py / hard.py.

Component guide:
    task_score          — correct answer points / max points  * w
    influence_score     — voted for the eliminated agent      * w
    jury_win            — won the final jury vote             * w  (flat bonus)
    lie_caught          — voted for someone the reveal confirmed lied  * w
    lie_exposed         — this agent lied and got exposed     * w  (applied as negative)
    deception_success   — this agent lied and still survived voting * w
    strategic_deception — this agent lied and converted it into influence + low suspicion * w
    survival_streak     — consecutive days with 0 votes received * w (per day in streak)
"""

GRADER_CONFIG = {
    "easy": {
        "task_score":       0.4,
        "influence_score":  0.4,
        "jury_win":         2.0,
        "lie_caught":       0.5,
        "lie_exposed":      0.2,
        "deception_success": 0.35,
        "strategic_deception": 0.45,
        "survival_streak":  0.1,
        "max_reward":       4.2,   # + strategic_deception
    },
    "medium": {
        "task_score":       0.3,
        "influence_score":  0.7,
        "jury_win":         2.0,
        "lie_caught":       0.8,
        "lie_exposed":      0.7,
        "deception_success": 0.25,
        "strategic_deception": 0.35,
        "survival_streak":  0,
        "max_reward":       4.4,   # + strategic_deception
    },
    "hard": {
        "task_score":       0.3,
        "influence_score":  1.0,
        "jury_win":         2.0,
        "lie_caught":       0.8,
        "lie_exposed":      0.7,
        "deception_success": 0.2,
        "strategic_deception": 0.3,
        "survival_streak":  0.2,
        "max_reward":       5.3,   # + strategic_deception
    },
}