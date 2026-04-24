"""
PM Environment

# Methods to write
The environment needs to do three things — advance the phase, collect actions from agents, and produce observations. So the core class probably has methods organized around those three responsibilities.
Think of it in terms of what happens at each phase transition:
Phase 1 → 2: env deals private info to each agent, broadcasts public info, then waits for each agent to submit their PreTaskMessage.
Phase 2 → 3: env collects all pre-task messages, validates one-per-agent, then runs the task and produces a TaskResult.
Phase 3 → 4: env publishes the task result, opens up Phase 4 message passing with the 5-message-per-side cap enforced.
Phase 4 → 5: env collects all trust assessments, applies them to the trust matrix, builds the DayPublicReveal, triggers the end-of-day summary LLM call per agent.
Phase 5 → next day or game over: env collects votes, resolves elimination, checks is_game_over, either resets for next day or ends.
So the skeleton is roughly:

reset() — initializes state, deals starting info
step(action: PMAction) — the main method, routes action to correct phase handler
_handle_phase_X(action) — one private method per phase
_advance_phase() — moves PMState.phase forward, triggers any phase-entry logic
get_observation(agent_id) — builds PMObservation for a specific agent
_build_public_reveal() — computes DayPublicReveal from the day's messages
_generate_day_summary(agent_id) — triggers the LLM summary call

Give that a shot and share what you write.
"""

from typing import Any, Optional, List
from uuid import uuid4
import numpy as np
import random
import matplotlib
from itertools import combinations
import copy
import base64
from PIL import Image
import io

from openenv.core.env_server import Environment

try:
    from ..models import PMState, PMObservation, PMState
    from .config import *
    from .utils import PMState, PMObservation, PMState
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    from models import PMState, PMObservation, PMState
    from server.config import *
    from server.utils import PMState, PMObservation, PMState

class PMEnvironment(Environment):
    """
    """
    SUPPORTS_CONCURRENT_SESSIONS = True
    def __init__(self,):
        super().__init__()

        self.is_done = False
        self.day = 0
        self.phase = 1
    
    def reset(self, task=None):
        if task is None:
            self.task = random.choice(['easy', 'medium', 'hard'])
        else:
            normalized_task = task.replace("task_", "")
            assert normalized_task in ['easy', 'medium', 'hard'], f"task value '{task}' must be from ['easy', 'medium', 'hard'] or prefixed with 'task_'"
            self.task = normalized_task
        
        self.task_config = TASK_CONFIGS[self.task]

        self.day = 0
        self.phase = 

    def step(Self,):
        pass
    
    def 
