"""
inference.py — Project Machiavelli
====================================
Thin accessor wrappers over PMEnvironment.global_inference_store.

These helpers are the intended interface for training loops / LLM agents
that need to read cross-episode learning signals without reaching into the
environment internals directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from server.environment import PMEnvironment


# ---------------------------------------------------------------------------
# Prior snapshots
# ---------------------------------------------------------------------------

def get_agent_prior(env: PMEnvironment, agent_id: int) -> Optional[Dict[str, Any]]:
    """
    Return the latest AgentPriorSnapshot for agent_id as a plain dict.

    Keys: agent_id, episode_index, truthful_prior, deception_prior,
          risk_beta, final_trust_scores.

    Returns None if no episode has completed yet for this agent.
    """
    return env.get_agent_prior(agent_id)


def get_all_agent_priors(env: PMEnvironment) -> Dict[int, Dict[str, Any]]:
    """Return latest prior snapshots for ALL agents seen so far."""
    store = env.global_inference_store
    return {
        aid: snap.model_dump()
        for aid, snap in store.agent_priors.items()
    }


# ---------------------------------------------------------------------------
# Episode records
# ---------------------------------------------------------------------------

def get_episode_record(env: PMEnvironment, episode_index: int) -> Optional[Dict[str, Any]]:
    """
    Return the full EpisodeRecord for a specific episode as a plain dict.

    Keys: episode_index, task, n_agents, days_played, winner_ids,
          eliminated_order, agent_day_summaries, prior_snapshots.

    agent_day_summaries[agent_id] is a list of day_summary dicts (only days
    the agent was alive), each with pre_discussion / task / post_discussion /
    voting sub-dicts.
    """
    return env.get_episode_record(episode_index)


def get_all_episode_records(env: PMEnvironment) -> List[Dict[str, Any]]:
    """Return all completed episode records as a list of plain dicts."""
    return [rec.model_dump() for rec in env.global_inference_store.episodes]


# ---------------------------------------------------------------------------
# Won-episode tracking
# ---------------------------------------------------------------------------

def get_winner_episodes(env: PMEnvironment, agent_id: int) -> List[int]:
    """
    Return the list of episode indices that agent_id survived / won.
    Empty list if the agent has never won an episode.
    """
    return env.get_winner_episodes(agent_id)


def get_all_winner_episodes(env: PMEnvironment) -> Dict[int, List[int]]:
    """Return { agent_id: [episode_indices won] } for every agent."""
    return dict(env.global_inference_store.agent_won_episodes)


# ---------------------------------------------------------------------------
# Convenience: full global store as a plain dict
# ---------------------------------------------------------------------------

def get_global_inference_store(env: PMEnvironment) -> Dict[str, Any]:
    """Serialise the entire GlobalInferenceStore to a plain dict."""
    return env.global_inference_store.model_dump()
