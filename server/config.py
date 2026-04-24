"""
Configuration for Project Machiavelli.
"""

GAME_CONFIGS = {
    "easy": {
        # agents
        "n_agents":          4,     # FIXED

        # game
        "task_type":         "individual",

        # task — MCQ private info options
        "n_options":         1,       # agent gets the correct answer directly

        # phase 4
        "max_post_discussion_messages": 5,

        # scoring
        "correct_answer_points":   10,
        "lie_penalty":              0,
        "lie_acknowledged_penalty": 0,

    },
    "medium": {
        # agents
        "n_agents":          4,     # FIXED

        # game
        "task_type":         "individual",

        # task
        "n_options":         2,       # agent gets 2 options, one correct

        # phase 4
        "max_post_discussion_messages": 5,

        # scoring
        "correct_answer_points":   10,
        "lie_penalty":              2,
        "lie_acknowledged_penalty": 1,

    },
    "hard": {
        # agents
        "n_agents":          4,

        # game
        "task_type":         "both",

        # task
        "n_options":         3,       # agent gets 3 options, one correct

        # phase 4
        "max_post_discussion_messages": 10,

        # scoring
        "correct_answer_points":   10,
        "lie_penalty":              5,
        "lie_acknowledged_penalty": 2,

    },
}