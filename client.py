"""
Project Machiavelli — Environment Client

Connects to the PMEnvironment server via WebSocket for persistent sessions.

Usage:
    from client import PMEnv
    from models import PMAction, ActionType, PreTaskMessage, MessageVeracity

    env = PMEnv(url="ws://localhost:7860")
    obs_map = env.reset(task="easy")

    # Submit a Phase 2 message for agent 0
    action = PMAction(
        agent_id=0,
        action_type=ActionType.SEND_PRE_TASK_MESSAGE,
        pre_task_message=PreTaskMessage(
            sender_id=0,
            recipient_id=1,
            content="I know the answer is 42.",
            veracity=MessageVeracity.TRUTH,
            day=1,
        ),
    )
    result = env.step(action)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

try:
    from ..models import (
        PMAction, PMObservation, PMState,
        Phase, TaskType, TaskResult,
        PreTaskMessage, PostDiscussionMessage,
        TrustAssessment, TrustDelta, MessageVeracity,
        DayPublicReveal, VoteRecord, DayHistoryEntry,
        ActionType,
    )
except ImportError:
    from models import (
        PMAction, PMObservation, PMState,
        Phase, TaskType, TaskResult,
        PreTaskMessage, PostDiscussionMessage,
        TrustAssessment, TrustDelta, MessageVeracity,
        DayPublicReveal, VoteRecord, DayHistoryEntry,
        ActionType,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_pre_task_message(d: Dict[str, Any]) -> PreTaskMessage:
    return PreTaskMessage(
        sender_id=d["sender_id"],
        recipient_id=d.get("recipient_id"),
        content=d["content"],
        veracity=MessageVeracity(d["veracity"]),
        day=d["day"],
        private_info_referenced=d.get("private_info_referenced"),
    )


def _parse_post_discussion_message(d: Dict[str, Any]) -> PostDiscussionMessage:
    return PostDiscussionMessage(
        sender_id=d["sender_id"],
        recipient_id=d["recipient_id"],
        content=d["content"],
        day=d["day"],
        turn_index=d["turn_index"],
    )


def _parse_task_result(d: Optional[Dict[str, Any]]) -> Optional[TaskResult]:
    if not d:
        return None
    return TaskResult(
        day=d["day"],
        task_type=TaskType(d["task_type"]),
        per_agent_outcome=d["per_agent_outcome"],
        ground_truth_exposed=d.get("ground_truth_exposed"),
        collective_success=d.get("collective_success"),
    )


def _parse_public_reveal(d: Optional[Dict[str, Any]]) -> Optional[DayPublicReveal]:
    if not d:
        return None
    return DayPublicReveal(
        day=d["day"],
        lies_told=d["lies_told"],
        lies_acknowledged=d["lies_acknowledged"],
        lies_unacknowledged=d["lies_unacknowledged"],
        task_scores=d["task_scores"],
    )


def _parse_observation(obs_data: Dict[str, Any]) -> PMObservation:
    return PMObservation(
        day=obs_data.get("day", 0),
        phase=Phase(obs_data.get("phase", "task_reveal")),
        alive_agents=obs_data.get("alive_agents", []),
        own_private_info=obs_data.get("own_private_info", ""),
        global_public_info=obs_data.get("global_public_info", ""),
        own_points=obs_data.get("own_points", 0),
        trust_scores=obs_data.get("trust_scores", {}),
        pre_task_messages_received=[
            _parse_pre_task_message(m)
            for m in obs_data.get("pre_task_messages_received", [])
        ],
        latest_task_result=_parse_task_result(obs_data.get("latest_task_result")),
        post_discussion_thread=[
            _parse_post_discussion_message(m)
            for m in obs_data.get("post_discussion_thread", [])
        ],
        public_reveal=_parse_public_reveal(obs_data.get("public_reveal")),
        revealed_veracity_map={
            int(k): MessageVeracity(v)
            for k, v in obs_data.get("revealed_veracity_map", {}).items()
        },
        votes_last_round={
            int(k): int(v)
            for k, v in obs_data.get("votes_last_round", {}).items()
        },
    )


def _parse_pm_state(payload: Dict[str, Any]) -> PMState:
    return PMState(
        day=payload.get("day", 0),
        phase=Phase(payload.get("phase", "task_reveal")),
        alive_agents=payload.get("alive_agents", []),
        task_type=TaskType(payload.get("task_type", "individual")),
        task_rules=payload.get("task_rules", ""),
        each_agent_private_info=payload.get("each_agent_private_info", {}),
        global_public_info=payload.get("global_public_info", ""),
        pre_task_messages={},       # wire up if needed for full state replay
        task_results={},
        post_discussion_messages={},
        trust_assessments={},
        trust_scores_dict={
            int(k): {int(kk): float(vv) for kk, vv in v.items()}
            for k, v in payload.get("trust_scores_dict", {}).items()
        },
        public_reveals={},
        vote_history=[
            VoteRecord(
                day=vr["day"],
                votes_cast={int(k): int(v) for k, v in vr["votes_cast"].items()},
                vote_counts={int(k): int(v) for k, v in vr["vote_counts"].items()},
                eliminated_id=vr.get("eliminated_id"),
                was_tie=vr.get("was_tie", False),
            )
            for vr in payload.get("vote_history", [])
        ],
        agents_point_map={
            int(k): int(v) for k, v in payload.get("agents_point_map", {}).items()
        },
        agent_removed_dict={
            int(k): int(v) for k, v in payload.get("agent_removed_dict", {}).items()
        },
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class PMEnv(EnvClient[PMAction, PMObservation, PMState]):
    """
    WebSocket client for the Project Machiavelli environment.

    The server sends back a Dict[agent_id, PMObservation] on every step.
    The client exposes the full map via result.observation (keyed by agent id)
    and also stores it on self.last_obs_map for convenience.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Stores the latest {agent_id: PMObservation} map after each step/reset
        self.last_obs_map: Dict[int, PMObservation] = {}
        self.last_rewards: Dict[int, float] = {}
        self.last_done: bool = False

    # ------------------------------------------------------------------
    # _step_payload  — serialize PMAction → wire dict
    # ------------------------------------------------------------------

    def _step_payload(self, action: PMAction) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "agent_id":    action.agent_id,
            "action_type": action.action_type.value,
        }

        if action.action_type == ActionType.SEND_PRE_TASK_MESSAGE:
            msg = action.pre_task_message
            payload["pre_task_message"] = {
                "sender_id":               msg.sender_id,
                "recipient_id":            msg.recipient_id,
                "content":                 msg.content,
                "veracity":                msg.veracity.value,
                "day":                     msg.day,
                "private_info_referenced": msg.private_info_referenced,
            }

        elif action.action_type == ActionType.SUBMIT_TASK_INPUT:
            payload["task_input"] = action.task_input

        elif action.action_type == ActionType.SEND_POST_DISCUSSION_MSG:
            msg = action.post_discussion_msg
            payload["post_discussion_msg"] = {
                "sender_id":    msg.sender_id,
                "recipient_id": msg.recipient_id,
                "content":      msg.content,
                "day":          msg.day,
                "turn_index":   msg.turn_index,
            }

        elif action.action_type == ActionType.SUBMIT_TRUST_ASSESSMENT:
            ta = action.trust_assessment
            payload["trust_assessment"] = {
                "assessor_id": ta.assessor_id,
                "target_id":   ta.target_id,
                "day":         ta.day,
                "reasoning":   ta.reasoning,
                "delta":       ta.delta.value,
            }

        elif action.action_type == ActionType.VOTE:
            payload["vote_target"] = action.vote_target

        return payload

    # ------------------------------------------------------------------
    # _parse_result  — wire dict → StepResult[PMObservation]
    # ------------------------------------------------------------------

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PMObservation]:
        """
        The server returns observations keyed by agent_id.
        We parse the full map and store it; StepResult gets agent 0's obs
        (or the first alive agent) as the primary observation.
        """
        raw_obs_map: Dict[str, Any] = payload.get("observation", {})

        obs_map: Dict[int, PMObservation] = {
            int(agent_id): _parse_observation(obs_data)
            for agent_id, obs_data in raw_obs_map.items()
        }
        self.last_obs_map = obs_map
        self.last_rewards = {
            int(k): float(v)
            for k, v in payload.get("rewards", {}).items()
        }
        self.last_done = payload.get("done", False)

        # Primary observation for StepResult — agent 0 or first available
        primary_obs = obs_map.get(0) or next(iter(obs_map.values()), None)

        return StepResult(
            observation=primary_obs,
            reward=payload.get("reward", None),
            done=self.last_done,
        )

    # ------------------------------------------------------------------
    # _parse_state  — wire dict → PMState
    # ------------------------------------------------------------------

    def _parse_state(self, payload: Dict[str, Any]) -> PMState:
        return _parse_pm_state(payload)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def obs_for(self, agent_id: int) -> Optional[PMObservation]:
        """Get the latest observation for a specific agent."""
        return self.last_obs_map.get(agent_id)

    def reward_for(self, agent_id: int) -> float:
        """Get the latest reward for a specific agent."""
        return self.last_rewards.get(agent_id, 0.0)

    @property
    def alive_agents(self) -> List[int]:
        """List of currently alive agent IDs from the latest observation."""
        if not self.last_obs_map:
            return []
        obs = next(iter(self.last_obs_map.values()))
        return obs.alive_agents

    @property
    def current_phase(self) -> Optional[Phase]:
        """Current phase from the latest observation."""
        if not self.last_obs_map:
            return None
        obs = next(iter(self.last_obs_map.values()))
        return obs.phase

    @property
    def current_day(self) -> Optional[int]:
        """Current day from the latest observation."""
        if not self.last_obs_map:
            return None
        obs = next(iter(self.last_obs_map.values()))
        return obs.day