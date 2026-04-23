"""
Data models for Project Machiavelli.

Day structure:
  Phase 1 — TASK_REVEAL:      public info broadcast, private info dealt per agent
  Phase 2 — PRE_DISCUSSION:   each agent sends 1 message (truth / twist / lie)
  Phase 3 — TASK_EXECUTION:   task performed, results published
  Phase 4 — POST_DISCUSSION:  agents talk (max 5 messages per pair), trust updates
  Phase 5 — VOTING:           silent simultaneous vote, one agent eliminated
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Phase(str, Enum):
    TASK_REVEAL      = "task_reveal"
    PRE_DISCUSSION   = "pre_discussion"
    TASK_EXECUTION   = "task_execution"
    POST_DISCUSSION  = "post_discussion"
    VOTING           = "voting"


class TaskType(str, Enum):
    INDIVIDUAL = "individual"   # each agent acts independently, scores are solo
    COALITION  = "coalition"    # agents pool effort, score is collective


class MessageVeracity(str, Enum):
    """
    The hidden label the sending agent assigns to their own Phase 2 message.
    The environment stores it but does not expose it until the Phase 5 public reveal.
    """
    TRUTH = "truth"   # private info shared faithfully
    TWIST = "twist"   # private info shared with a plausible distortion
    LIE   = "lie"     # private info fabricated or inverted


class TrustDelta(str, Enum):
    """
    Qualitative trust change an agent submits after Phase 4 discussion.
    Converted to a float when applied to the trust matrix.
    """
    STRONG_INCREASE = "strong_increase"   # +0.20
    INCREASE        = "increase"          # +0.10
    NEUTRAL         = "neutral"           # +0.00
    DECREASE        = "decrease"          # -0.10
    STRONG_DECREASE = "strong_decrease"   # -0.20

    def to_float(self) -> float:
        return {
            "strong_increase":  0.20,
            "increase":         0.10,
            "neutral":          0.00,
            "decrease":        -0.10,
            "strong_decrease": -0.20,
        }[self.value]


# ---------------------------------------------------------------------------
# Phase 2 — Pre-task message
# ---------------------------------------------------------------------------

class PreTaskMessage(BaseModel):
    """
    The single message each agent sends in Phase 2.
    veracity is self-labeled by the agent's LLM — the environment does not
    verify it. recipient_id=None means broadcast to all alive agents.
    veracity is hidden from all observers until the Phase 5 public reveal.
    """
    model_config = ConfigDict(frozen=True)

    sender_id:               int
    recipient_id:            Optional[int]     # None = broadcast
    content:                 str
    veracity:                MessageVeracity   # self-labeled by the agent
    day:                     int
    private_info_referenced: Optional[str] = None  # snapshot of info the agent based this on

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message content cannot be empty.")
        return v

    @property
    def is_private(self) -> bool:
        return self.recipient_id is not None


# ---------------------------------------------------------------------------
# Phase 4 — Post-task discussion
# ---------------------------------------------------------------------------

class PostDiscussionMessage(BaseModel):
    """
    One message in the Phase 4 back-and-forth between two agents.
    Max 5 messages per side per pair per day (enforced by the environment).
    Agents can deny, confront, or acknowledge lies here.
    """
    model_config = ConfigDict(frozen=True)

    sender_id:    int
    recipient_id: int       # always directed — no broadcasts in Phase 4
    content:      str
    day:          int
    turn_index:   int       # 0-indexed position within this pair's conversation

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message content cannot be empty.")
        return v


class TrustAssessment(BaseModel):
    """
    After Phase 4 discussion, each agent submits one assessment per peer.
    The env applies delta to the trust score matrix immediately.
    reasoning is logged for analysis but never shared with other agents.
    """
    model_config = ConfigDict(frozen=True)

    assessor_id: int
    target_id:   int
    day:         int
    reasoning:   str         # LLM chain-of-thought, private
    delta:       TrustDelta


# ---------------------------------------------------------------------------
# Phase 3 — Task result
# ---------------------------------------------------------------------------

class TaskResult(BaseModel):
    """
    Published at end of Phase 3. Visible to all alive agents immediately.
    ground_truth_exposed is optional env-revealed info agents can use
    in Phase 4 to evaluate whether they were lied to.
    """
    model_config = ConfigDict(frozen=True)

    day:                  int
    task_type:            TaskType
    per_agent_outcome:    Dict[int, int]    # agent_id → points earned this round
    ground_truth_exposed: Optional[str] = None
    collective_success:   Optional[bool] = None  # relevant for coalition tasks only


# ---------------------------------------------------------------------------
# Phase 5 — Public reveal and voting
# ---------------------------------------------------------------------------

class DayPublicReveal(BaseModel):
    """
    The structured block broadcast to all agents before voting.
    Contains full honesty stats per agent — identities revealed,
    but not who each lie was directed at.

    lies_told:           total lies sent by each agent that day
    lies_acknowledged:   how many of those they admitted during Phase 4
    lies_unacknowledged: lies that survived Phase 4 without confession
    task_scores:         points each agent earned in Phase 3
    """
    model_config = ConfigDict(frozen=True)

    day:                  int
    lies_told:            Dict[int, int]   # agent_id → count
    lies_acknowledged:    Dict[int, int]   # agent_id → count
    lies_unacknowledged:  Dict[int, int]   # agent_id → count
    task_scores:          Dict[int, int]   # agent_id → points

    @model_validator(mode="after")
    def unacknowledged_consistent(self) -> "DayPublicReveal":
        for agent_id in self.lies_told:
            told   = self.lies_told.get(agent_id, 0)
            acked  = self.lies_acknowledged.get(agent_id, 0)
            unacked = self.lies_unacknowledged.get(agent_id, 0)
            if acked + unacked != told:
                raise ValueError(
                    f"Agent {agent_id}: acknowledged ({acked}) + unacknowledged "
                    f"({unacked}) != lies_told ({told})."
                )
        return self


class VoteRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    day:           int
    votes_cast:    Dict[int, int]   # voter_id → target_id (simultaneous, silent)
    vote_counts:   Dict[int, int]   # target_id → votes received
    eliminated_id: Optional[int]    # None on an unresolved tie
    was_tie:       bool = False


# ---------------------------------------------------------------------------
# Day history entry — one per agent per day
# ---------------------------------------------------------------------------

class DayHistoryEntry(BaseModel):
    """
    Stored in each agent's chat history at end of day.
    The env passes objective_context (public reveal + agent's own Phase 4
    threads + their trust scores) to a single LLM call.
    The LLM writes summary_and_reflection, which becomes the history entry.
    """
    model_config = ConfigDict(frozen=True)

    day:                    int
    agent_id:               int
    objective_context:      str   # env-assembled facts passed as LLM input
    summary_and_reflection: str   # LLM output — personalized narrative


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id:              int
    alive:           bool  = True
    truthful_prior:  float = Field(0.5, ge=0.0, le=1.0)
    deception_prior: float = Field(0.5, ge=0.0, le=1.0)
    risk_beta:       float = Field(1.0, gt=0.0)

    trust_scores:  Dict[int, float] = Field(default_factory=dict)

    private_info:  str = ""
    public_info:   str = ""
    system_prompt: str = ""

    points: int = 0

    # Keyed by day — enforces one pre-task message per agent per day
    pre_task_messages: Dict[int, PreTaskMessage] = Field(default_factory=dict)

    # Chat history — one entry per day, built at end of each day
    day_history: List[DayHistoryEntry] = Field(default_factory=list)

    def update_trust(self, other_id: int, delta: float) -> None:
        current = self.trust_scores.get(other_id, 0.5)
        self.trust_scores[other_id] = round(max(0.0, min(1.0, current + delta)), 4)

    def record_pre_task_message(self, msg: PreTaskMessage) -> None:
        if msg.day in self.pre_task_messages:
            raise ValueError(
                f"Agent {self.id} already sent a pre-task message on day {msg.day}."
            )
        self.pre_task_messages[msg.day] = msg

    def add_day_history(self, entry: DayHistoryEntry) -> None:
        self.day_history.append(entry)

    @property
    def history_summary(self) -> str:
        """Concatenated narrative history — passed as context to LLM calls."""
        return "\n\n".join(e.summary_and_reflection for e in self.day_history)


# ---------------------------------------------------------------------------
# Environment state
# ---------------------------------------------------------------------------

class PMState(BaseModel):
    """Full ground-truth state. Never sent raw to any agent."""

    day:   int   = 0
    phase: Phase = Phase.TASK_REVEAL

    alive_agents: List[int]
    task_type:    TaskType
    task_rules:   str

    # Phase 1
    each_agent_private_info: Dict[int, str] = Field(default_factory=dict)
    global_public_info:      str            = ""

    # Phase 2 — day → sender_id → message
    pre_task_messages: Dict[int, Dict[int, PreTaskMessage]] = Field(default_factory=dict)

    # Phase 3
    task_results: Dict[int, TaskResult] = Field(default_factory=dict)

    # Phase 4 — day → list of messages across all pairs
    post_discussion_messages: Dict[int, List[PostDiscussionMessage]] = Field(default_factory=dict)

    # Phase 4 — day → list of assessments
    trust_assessments: Dict[int, List[TrustAssessment]] = Field(default_factory=dict)

    # Full trust matrix (ground truth)
    trust_scores_dict: Dict[int, Dict[int, float]] = Field(default_factory=dict)

    # Phase 5
    public_reveals: Dict[int, DayPublicReveal] = Field(default_factory=dict)
    vote_history:   List[VoteRecord]           = Field(default_factory=list)

    # Scoring and elimination
    agents_point_map:   Dict[int, int] = Field(default_factory=dict)
    agent_removed_dict: Dict[int, int] = Field(default_factory=dict)  # agent_id → day removed

    @field_validator("alive_agents")
    @classmethod
    def at_least_one_agent(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("alive_agents cannot be empty at init.")
        return v

    @property
    def is_game_over(self) -> bool:
        return len(self.alive_agents) <= 1

    @property
    def votes_last_round(self) -> Dict[int, int]:
        return self.vote_history[-1].votes_cast if self.vote_history else {}

    def record_pre_task_message(self, msg: PreTaskMessage) -> None:
        day_msgs = self.pre_task_messages.setdefault(msg.day, {})
        if msg.sender_id in day_msgs:
            raise ValueError(
                f"Agent {msg.sender_id} already sent a pre-task message on day {msg.day}."
            )
        day_msgs[msg.sender_id] = msg

    def messages_visible_to(self, agent_id: int, day: int) -> List[PreTaskMessage]:
        """Phase 2 messages this agent can see — broadcast + messages addressed to them."""
        return [
            m for m in self.pre_task_messages.get(day, {}).values()
            if m.recipient_id is None or m.recipient_id == agent_id
        ]

    def post_discussion_thread(
        self, day: int, agent_a: int, agent_b: int
    ) -> List[PostDiscussionMessage]:
        """All Phase 4 messages exchanged between two specific agents on a given day."""
        return [
            m for m in self.post_discussion_messages.get(day, [])
            if {m.sender_id, m.recipient_id} == {agent_a, agent_b}
        ]

    def phase4_message_count(self, day: int, sender_id: int, recipient_id: int) -> int:
        """How many Phase 4 messages sender has sent to recipient today — enforces the cap."""
        return sum(
            1 for m in self.post_discussion_messages.get(day, [])
            if m.sender_id == sender_id and m.recipient_id == recipient_id
        )

    def apply_trust_assessment(self, assessment: TrustAssessment) -> None:
        self.trust_assessments.setdefault(assessment.day, []).append(assessment)
        agent_trust = self.trust_scores_dict.setdefault(assessment.assessor_id, {})
        current = agent_trust.get(assessment.target_id, 0.5)
        updated = max(0.0, min(1.0, current + assessment.delta.to_float()))
        agent_trust[assessment.target_id] = round(updated, 4)


# ---------------------------------------------------------------------------
# Per-agent observation
# ---------------------------------------------------------------------------

class PMObservation(BaseModel):
    """
    What one agent sees — constructed by the env from PMState.
    veracity labels on Phase 2 messages are hidden until reveal_veracity=True
    which the env only sets during Phase 5 (after the public reveal is built).
    """
    day:   int
    phase: Phase

    alive_agents:       List[int]
    own_private_info:   str
    global_public_info: str
    own_points:         int
    trust_scores:       Dict[int, float]

    # Phase 2
    pre_task_messages_received: List[PreTaskMessage] = Field(default_factory=list)

    # Phase 3
    latest_task_result: Optional[TaskResult] = None

    # Phase 4
    post_discussion_thread: List[PostDiscussionMessage] = Field(default_factory=list)

    # Phase 5 — veracity now visible, public reveal published
    public_reveal:         Optional[DayPublicReveal]          = None
    revealed_veracity_map: Dict[int, MessageVeracity]         = Field(default_factory=dict)

    # Voting
    votes_last_round: Dict[int, int] = Field(default_factory=dict)

    @classmethod
    def from_state(
        cls,
        state: PMState,
        agent_id: int,
        reveal_veracity: bool = False,
    ) -> "PMObservation":
        visible_msgs = state.messages_visible_to(agent_id, state.day)

        revealed: Dict[int, MessageVeracity] = {}
        if reveal_veracity:
            revealed = {m.sender_id: m.veracity for m in visible_msgs}

        return cls(
            day=state.day,
            phase=state.phase,
            alive_agents=state.alive_agents,
            own_private_info=state.each_agent_private_info.get(agent_id, ""),
            global_public_info=state.global_public_info,
            own_points=state.agents_point_map.get(agent_id, 0),
            trust_scores=state.trust_scores_dict.get(agent_id, {}),
            pre_task_messages_received=visible_msgs,
            latest_task_result=state.task_results.get(state.day),
            post_discussion_thread=state.post_discussion_messages.get(state.day, []),
            public_reveal=state.public_reveals.get(state.day),
            revealed_veracity_map=revealed,
            votes_last_round=state.votes_last_round,
        )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    SEND_PRE_TASK_MESSAGE    = "send_pre_task_message"     # Phase 2
    SUBMIT_TASK_INPUT        = "submit_task_input"         # Phase 3
    SEND_POST_DISCUSSION_MSG = "send_post_discussion_msg"  # Phase 4
    SUBMIT_TRUST_ASSESSMENT  = "submit_trust_assessment"   # Phase 4
    VOTE                     = "vote"                      # Phase 5


class PMAction(BaseModel):
    """One action from one agent. Env validates legality against current phase."""

    agent_id:    int
    action_type: ActionType

    pre_task_message:     Optional[PreTaskMessage]        = None  # Phase 2
    task_input:           Optional[str]                   = None  # Phase 3
    post_discussion_msg:  Optional[PostDiscussionMessage] = None  # Phase 4
    trust_assessment:     Optional[TrustAssessment]       = None  # Phase 4
    vote_target:          Optional[int]                   = None  # Phase 5

    @model_validator(mode="after")
    def validate_payload(self) -> "PMAction":
        t = self.action_type
        if t == ActionType.SEND_PRE_TASK_MESSAGE and self.pre_task_message is None:
            raise ValueError("SEND_PRE_TASK_MESSAGE requires pre_task_message.")
        if t == ActionType.SUBMIT_TASK_INPUT and self.task_input is None:
            raise ValueError("SUBMIT_TASK_INPUT requires task_input.")
        if t == ActionType.SEND_POST_DISCUSSION_MSG and self.post_discussion_msg is None:
            raise ValueError("SEND_POST_DISCUSSION_MSG requires post_discussion_msg.")
        if t == ActionType.SUBMIT_TRUST_ASSESSMENT and self.trust_assessment is None:
            raise ValueError("SUBMIT_TRUST_ASSESSMENT requires trust_assessment.")
        if t == ActionType.VOTE and self.vote_target is None:
            raise ValueError("VOTE requires vote_target.")
        return self