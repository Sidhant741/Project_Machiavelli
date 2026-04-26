"""
FastAPI + Gradio application for Project Machiavelli.

A multi-agent social survival environment where LLM agents compete
through deception, negotiation, and coalition formation.

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Direct:
    python app.py
"""

import os
import sys
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent.parent
for path in [repo_root, repo_root / "server", repo_root / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# ---------------------------------------------------------------------------
# Env imports
# ---------------------------------------------------------------------------
try:
    from ..models import (
        PMAction, PMObservation, ActionType, Phase,
        PreTaskMessage, PostDiscussionMessage,
        TrustAssessment, TrustDelta, MessageVeracity,
    )
    from .environment import PMEnvironment
    from .config import GAME_CONFIGS
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    from models import (
        PMAction, PMObservation, ActionType, Phase,
        PreTaskMessage, PostDiscussionMessage,
        TrustAssessment, TrustDelta, MessageVeracity,
    )
    from server.environment import PMEnvironment
    from server.config import GAME_CONFIGS

# ---------------------------------------------------------------------------
# OpenEnv web interface
# ---------------------------------------------------------------------------
from openenv.core.env_server import create_web_interface_app

def create_pm_environment():
    return PMEnvironment()

app = create_web_interface_app(
    create_pm_environment,
    PMAction,
    PMObservation,
    env_name="machiavelli",
    max_concurrent_envs=10,
    path="/",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)