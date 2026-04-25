"""
Data models for Project Machiavelli.

Day structure:
  Phase 1 — TASK_REVEAL:      public info broadcast, private info dealt per agent
  Phase 2 — PRE_DISCUSSION:   each agent sends 1 message to every other agent (truth / twist / lie)
  Phase 3 — TASK_EXECUTION:   task performed, results published
  Phase 4 — POST_DISCUSSION:  agents talk (max N messages per pair), trust updates
  Phase 5 — VOTING:           silent simultaneous vote, one agent eliminated

End-game (2 finalists remain):
  JURY_VOTE:  eliminated agents vote for winner based on full chat history
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
    JURY_VOTE        = "jury_vote"        # final phase — eliminated agents decide winner

class TaskType(str, Enum):
    INDIVIDUAL = "individual"
    COALITION  = "coalition"

class MessageVeracity(str, Enum):
    """
    The hidden label the sending agent assigns to their own Phase 2 message.
    Hidden from all observers until the Phase 5 public reveal.
    """
    TRUTH = "truth"
    TWIST = "twist"
    LIE   = "lie"


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
    One message sent in Phase 2.
    Each agent sends exactly one message to every other alive agent.
    veracity is self-labeled by the sending agent's LLM.
    veracity is hidden from all observers until the Phase 5 public reveal.
    """
    model_config = ConfigDict(frozen=True)

    sender_id:               int
    recipient_id:            int
    content:                 str
    veracity:                MessageVeracity
    day:                     int
    private_info_referenced: Optional[str] = None

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
    Max N messages per side per pair per day (enforced by the environment).
    """
    model_config = ConfigDict(frozen=True)

    sender_id:    int
    recipient_id: int
    content:      str
    day:          int
    turn_index:   int

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
    reasoning is logged for analysis but never shared with other agents.
    """
    model_config = ConfigDict(frozen=True)

    assessor_id: int
    target_id:   int
    day:         int
    reasoning:   str
    delta:       TrustDelta


# ---------------------------------------------------------------------------
# Phase 3 — Task result
# ---------------------------------------------------------------------------

class TaskResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    day:                  int
    task_type:            TaskType
    per_agent_outcome:    Dict[int, int]
    ground_truth_exposed: Optional[str] = None
    collective_success:   Optional[bool] = None


# ---------------------------------------------------------------------------
# Phase 5 — Public reveal and voting
# ---------------------------------------------------------------------------

class DayPublicReveal(BaseModel):
    """
    Broadcast to all agents before voting.
    Contains honesty stats per agent — identities revealed,
    but not who each lie was directed at.
    """
    model_config = ConfigDict(frozen=True)

    day:                  int
    lies_told:            Dict[int, int]
    lies_acknowledged:    Dict[int, int]
    lies_unacknowledged:  Dict[int, int]
    task_scores:          Dict[int, int]

    @model_validator(mode="after")
    def unacknowledged_consistent(self) -> "DayPublicReveal":
        for agent_id in self.lies_told:
            told    = self.lies_told.get(agent_id, 0)
            acked   = self.lies_acknowledged.get(agent_id, 0)
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
    votes_cast:    Dict[int, int]
    vote_counts:   Dict[int, int]
    eliminated_id: Optional[int]
    was_tie:       bool = False


# ---------------------------------------------------------------------------
# Jury vote — end-game only
# ---------------------------------------------------------------------------

class JuryBallot(BaseModel):
    """
    One eliminated agent's jury vote for a finalist.
    Generated by an LLM call using the juror's full day_history
    and the finalists' public chat records.
    """
    model_config = ConfigDict(frozen=True)

    juror_id:    int           # eliminated agent casting the vote
    vote_for:    int           # finalist agent_id they're voting for
    reasoning:   str           # LLM chain-of-thought — logged, not shared


class JuryVerdict(BaseModel):
    """
    Aggregated result of all jury ballots.
    Produced once all jurors have voted.
    """
    model_config = ConfigDict(frozen=True)

    finalist_a:     int
    finalist_b:     int
    ballots:        List[JuryBallot]
    votes_for_a:    int
    votes_for_b:    int
    winner_id:      int            # the agent who won the game
    was_jury_tie:   bool = False   # True if ballots were split evenly (tiebreak applied)

    @model_validator(mode="after")
    def votes_sum_to_ballots(self) -> "JuryVerdict":
        total = self.votes_for_a + self.votes_for_b
        if total != len(self.ballots):
            raise ValueError(
                f"votes_for_a ({self.votes_for_a}) + votes_for_b ({self.votes_for_b}) "
                f"!= len(ballots) ({len(self.ballots)})."
            )
        return self


# ---------------------------------------------------------------------------
# Day history entry — one per agent per day
# ---------------------------------------------------------------------------

class DayHistoryEntry(BaseModel):
    """
    Stored in each agent's chat history at end of day.
    summary_and_reflection is LLM-written narrative — becomes the agent's
    long-term memory and is passed to jury LLM calls.
    """
    model_config = ConfigDict(frozen=True)

    day:                    int
    agent_id:               int
    objective_context:      str
    summary_and_reflection: str


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

    # Keyed by (sender_id, recipient_id) tuple encoded as "s_r" string
    # — one pre-task message per sender-recipient pair per day
    pre_task_messages: Dict[int, PreTaskMessage] = Field(default_factory=dict)

    day_history: List[DayHistoryEntry] = Field(default_factory=list)

    def update_trust(self, other_id: int, delta: float) -> None:
        current = self.trust_scores.get(other_id, 0.5)
        self.trust_scores[other_id] = round(max(0.0, min(1.0, current + delta)), 4)

    def record_pre_task_message(self, msg: PreTaskMessage) -> None:
        """Store by day — agent-level record (one entry per day for simplicity)."""
        self.pre_task_messages[msg.day] = msg

    def add_day_history(self, entry: DayHistoryEntry) -> None:
        self.day_history.append(entry)

    @property
    def history_summary(self) -> str:
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

    # Phase 2 — day → (sender_id, recipient_id) encoded as "s__r" → message
    # Allows one message per sender-recipient pair per day
    pre_task_messages: Dict[int, Dict[str, PreTaskMessage]] = Field(default_factory=dict)

    # Phase 3
    task_results: Dict[int, TaskResult] = Field(default_factory=dict)

    # Phase 4
    post_discussion_messages: Dict[int, List[PostDiscussionMessage]] = Field(default_factory=dict)
    trust_assessments:        Dict[int, List[TrustAssessment]]       = Field(default_factory=dict)

    # Trust matrix (ground truth)
    trust_scores_dict: Dict[int, Dict[int, float]] = Field(default_factory=dict)

    # Phase 5
    public_reveals: Dict[int, DayPublicReveal] = Field(default_factory=dict)
    vote_history:   List[VoteRecord]           = Field(default_factory=list)

    # Jury vote end-game
    jury_verdict:                  Optional[JuryVerdict]              = None
    game_winner:                   Optional[int]                      = None
    # Snapshot of each eliminated agent's day_history at time of elimination
    eliminated_agents_history:     Dict[int, List[DayHistoryEntry]]   = Field(default_factory=dict)

    # Scoring and elimination
    agents_point_map:   Dict[int, int] = Field(default_factory=dict)
    agent_removed_dict: Dict[int, int] = Field(default_factory=dict)

    @field_validator("alive_agents")
    @classmethod
    def at_least_one_agent(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("alive_agents cannot be empty at init.")
        return v

    @property
    def is_game_over(self) -> bool:
        """Game ends when jury verdict is in (winner decided) or only 1 agent left."""
        return self.game_winner is not None or len(self.alive_agents) <= 1

    @property
    def votes_last_round(self) -> Dict[int, int]:
        return self.vote_history[-1].votes_cast if self.vote_history else {}

    # ------------------------------------------------------------------
    # Phase 2 helpers — pair-keyed messages
    # ------------------------------------------------------------------

    @staticmethod
    def _pair_key(sender_id: int, recipient_id: int) -> str:
        return f"{sender_id}__{recipient_id}"

    def record_pre_task_message(self, msg: PreTaskMessage) -> None:
        day_msgs = self.pre_task_messages.setdefault(msg.day, {})
        key = self._pair_key(msg.sender_id, msg.recipient_id)
        if key in day_msgs:
            raise ValueError(
                f"Agent {msg.sender_id} already sent a message to "
                f"Agent {msg.recipient_id} on day {msg.day}."
            )
        day_msgs[key] = msg

    def messages_visible_to(self, agent_id: int, day: int) -> List[PreTaskMessage]:
        """Phase 2 messages addressed to this agent."""
        return [
            m for m in self.pre_task_messages.get(day, {}).values()
            if m.recipient_id == agent_id
        ]

    def all_pre_task_messages_for_day(self, day: int) -> List[PreTaskMessage]:
        """All Phase 2 messages sent on a given day (env/host use only)."""
        return list(self.pre_task_messages.get(day, {}).values())

    # ------------------------------------------------------------------
    # Phase 4 helpers
    # ------------------------------------------------------------------

    def post_discussion_thread(
        self, day: int, agent_a: int, agent_b: int
    ) -> List[PostDiscussionMessage]:
        return [
            m for m in self.post_discussion_messages.get(day, [])
            if {m.sender_id, m.recipient_id} == {agent_a, agent_b}
        ]

    def phase4_message_count(self, day: int, sender_id: int, recipient_id: int) -> int:
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

    # ------------------------------------------------------------------
    # Jury helpers
    # ------------------------------------------------------------------

    def snapshot_eliminated_agent(self, agent_id: int, history: List[DayHistoryEntry]) -> None:
        """Called at elimination time — freeze this agent's history for jury use."""
        self.eliminated_agents_history[agent_id] = list(history)

    def finalist_chat_history(self, finalist_id: int) -> str:
        """
        All post-discussion messages sent or received by a finalist across all days.
        Passed to jury LLM prompts as evidence of social behaviour.
        """
        lines = []
        for day, msgs in self.post_discussion_messages.items():
            for m in msgs:
                if m.sender_id == finalist_id or m.recipient_id == finalist_id:
                    lines.append(
                        f"Day {day} | Agent {m.sender_id} → Agent {m.recipient_id}: {m.content}"
                    )
        return "\n".join(lines) if lines else "(no post-discussion messages recorded)"

    def finalist_public_stats(self, finalist_id: int) -> str:
        """
        Per-day lies and task scores for a finalist — from public reveals.
        """
        lines = []
        for day, reveal in self.public_reveals.items():
            fid_str = str(finalist_id)
            lies    = reveal.lies_told.get(finalist_id, reveal.lies_told.get(fid_str, 0))
            score   = reveal.task_scores.get(finalist_id, reveal.task_scores.get(fid_str, 0))
            lines.append(f"Day {day}: task score={score}, lies told={lies}")
        return "\n".join(lines) if lines else "(no public reveal data)"


# ---------------------------------------------------------------------------
# Per-agent observation
# ---------------------------------------------------------------------------

class PMObservation(BaseModel):
    """
    What one agent sees — constructed by the env from PMState.
    veracity labels on Phase 2 messages are hidden until reveal_veracity=True.
    """
    day:   int
    phase: Phase

    alive_agents:       List[int]
    own_private_info:   str
    global_public_info: str
    own_points:         int
    trust_scores:       Dict[int, float]

    # Phase 2 — messages this agent received
    pre_task_messages_received: List[PreTaskMessage] = Field(default_factory=list)

    # Phase 3
    latest_task_result: Optional[TaskResult] = None

    # Phase 4
    post_discussion_thread: List[PostDiscussionMessage] = Field(default_factory=list)

    # Phase 5
    public_reveal:         Optional[DayPublicReveal]  = None
    revealed_veracity_map: Dict[int, MessageVeracity] = Field(default_factory=dict)

    # Voting
    votes_last_round: Dict[int, int] = Field(default_factory=dict)

    # Jury / end-game
    jury_verdict: Optional[JuryVerdict] = None
    game_winner:  Optional[int]         = None

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
            jury_verdict=state.jury_verdict,
            game_winner=state.game_winner,
        )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    SEND_PRE_TASK_MESSAGE    = "send_pre_task_message"
    SUBMIT_TASK_INPUT        = "submit_task_input"
    SEND_POST_DISCUSSION_MSG = "send_post_discussion_msg"
    SUBMIT_TRUST_ASSESSMENT  = "submit_trust_assessment"
    VOTE                     = "vote"


class PMAction(BaseModel):
    """One action from one agent. Env validates legality against current phase."""

    agent_id:    int
    action_type: ActionType

    pre_task_message:     Optional[PreTaskMessage]        = None
    task_input:           Optional[str]                   = None
    post_discussion_msg:  Optional[PostDiscussionMessage] = None
    trust_assessment:     Optional[TrustAssessment]       = None
    vote_target:          Optional[int]                   = None

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


# ---------------------------------------------------------------------------
# Episode-level compression models  (GlobalInferenceStore)
# ---------------------------------------------------------------------------

class AgentPriorSnapshot(BaseModel):
    """
    Snapshot of an agent's prior beliefs captured at the END of an episode.
    These are carried into future episodes so the training loop can initialise
    the next episode's priors from observed behaviour.

    Fields
    ------
    agent_id        : which agent
    episode_index   : 0-indexed episode number this snapshot came from
    truthful_prior  : Agent.truthful_prior at game end
    deception_prior : Agent.deception_prior at game end
    risk_beta       : Agent.risk_beta at game end
    final_trust_scores : agent's outgoing trust map at game end  { peer_id: float }
    """
    model_config = ConfigDict(frozen=True)

    agent_id:           int
    episode_index:      int
    truthful_prior:     float = Field(ge=0.0, le=1.0)
    deception_prior:    float = Field(ge=0.0, le=1.0)
    risk_beta:          float = Field(gt=0.0)
    final_trust_scores: Dict[int, float] = Field(default_factory=dict)


class EpisodeRecord(BaseModel):
    """
    One record per completed episode stored in GlobalInferenceStore.

    Fields
    ------
    episode_index   : 0-indexed counter auto-incremented per reset()
    task            : difficulty string — "easy" | "medium" | "hard"
    n_agents        : number of agents the episode started with
    days_played     : total days completed before game over
    winner_ids      : agent IDs alive at game end (survivors / winners)
    eliminated_order: agent IDs in order of elimination (first eliminated first)
    agent_day_summaries : per-agent list of day-level summary dicts, only for
                          days the agent was alive.  { agent_id: [day_summary, ...] }
    prior_snapshots : prior beliefs for every agent at game end { agent_id: snapshot }
    """
    model_config = ConfigDict(frozen=True)

    episode_index:    int
    task:             str
    n_agents:         int
    days_played:      int
    winner_ids:       List[int]
    eliminated_order: List[int] = Field(default_factory=list)

    # { agent_id: [ day_summary_dict, ... ] }  — only days the agent was alive
    agent_day_summaries: Dict[int, List[Dict]] = Field(default_factory=dict)

    # { agent_id: AgentPriorSnapshot }
    prior_snapshots: Dict[int, AgentPriorSnapshot] = Field(default_factory=dict)


class GlobalInferenceStore(BaseModel):
    """
    Persists across reset() calls on PMEnvironment.
    Acts as the top-level container for all episode-level learning signals.

    Fields
    ------
    episodes        : ordered list of EpisodeRecord objects (one per completed game)
    agent_priors    : latest AgentPriorSnapshot per agent_id
                      (updated each episode — most-recent snapshot wins)
    agent_won_episodes: { agent_id: [episode_index, ...] }
                        list of episode indices each agent survived / won
    """
    episodes:             List[EpisodeRecord]          = Field(default_factory=list)
    agent_priors:         Dict[int, AgentPriorSnapshot] = Field(default_factory=dict)
    agent_won_episodes:   Dict[int, List[int]]          = Field(default_factory=dict)

    def record_episode(self, record: EpisodeRecord) -> None:
        """Append a completed episode and update running indexes."""
        self.episodes.append(record)
        # Update latest prior per agent
        for agent_id, snap in record.prior_snapshots.items():
            self.agent_priors[agent_id] = snap
        # Track which episodes each agent won
        for winner_id in record.winner_ids:
            self.agent_won_episodes.setdefault(winner_id, []).append(record.episode_index)

    def get_won_episodes(self, agent_id: int) -> List[int]:
        """Return list of episode indices agent_id survived/won."""
        return self.agent_won_episodes.get(agent_id, [])

    def get_latest_prior(self, agent_id: int) -> Optional[AgentPriorSnapshot]:
        """Return most-recent prior snapshot for agent_id, or None."""
        return self.agent_priors.get(agent_id)