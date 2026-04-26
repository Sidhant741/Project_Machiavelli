"""
phases.py — Project Machiavelli
=================================
All phase logic and phase-transition logic lives here.
PMEnvironment imports and delegates to these functions/classes.

Phase flow per day:
  Phase 1 — TASK_REVEAL      : private + public info dealt, no agent action
  Phase 2 — PRE_DISCUSSION   : each alive agent sends exactly 1 message
  Phase 3 — TASK_EXECUTION   : each alive agent submits their answer
  Phase 4 — POST_DISCUSSION  : up to N messages per pair + trust assessments
  Phase 5 — VOTING           : silent simultaneous vote → elimination
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

try:
    from ..models import (
        Agent, PMState, PMAction, PMObservation,
        ActionType, Phase, TaskType, TaskResult,
        PreTaskMessage, PostDiscussionMessage,
        TrustAssessment, TrustDelta, MessageVeracity,
        DayPublicReveal, VoteRecord,
    )
    from .config import GAME_CONFIGS
except ImportError:
    from models import (
        Agent, PMState, PMAction, PMObservation,
        ActionType, Phase, TaskType, TaskResult,
        PreTaskMessage, PostDiscussionMessage,
        TrustAssessment, TrustDelta, MessageVeracity,
        DayPublicReveal, VoteRecord,
    )
    from server.config import GAME_CONFIGS  # noqa: F401 (unused but keeps fallback consistent)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_question(
    n_options: int, day: int, agent_id: int
) -> Tuple[str, str, List[str]]:
    """
    Generate (question_text, correct_answer, all_options).
    Replace with a real trivia DB in production.
    """
    correct = str(random.randint(1, 100))
    distractors: List[str] = []
    while len(distractors) < n_options - 1:
        d = str(random.randint(1, 100))
        if d != correct and d not in distractors:
            distractors.append(d)
    options = [correct] + distractors
    random.shuffle(options)
    question = (
        f"Day {day} | Agent {agent_id}: "
        f"Guess the secret number. Options: {', '.join(options)}"
    )
    return question, correct, options


def _make_shared_question(
    n_options: int, day: int, n_agents: int
) -> Tuple[str, str, List[str]]:
    """
    Generate ONE shared question for the full day.
    All agents see the same question context, but each agent receives a different
    private option/hint extracted from the shared options list.
    """
    pool_size = max(n_options, n_agents, 2)
    correct = str(random.randint(1, 100))
    distractors: List[str] = []
    while len(distractors) < pool_size - 1:
        d = str(random.randint(1, 100))
        if d != correct and d not in distractors:
            distractors.append(d)
    options = [correct] + distractors
    random.shuffle(options)
    question = (
        f"Day {day}: Shared task for all agents — guess the secret number. "
        f"Candidate options: {', '.join(options)}"
    )
    return question, correct, options


def _resolve_tie(
    vote_counts: Dict[int, int], alive: List[int]
) -> Tuple[Optional[int], bool]:
    """Return (eliminated_id, was_tie). Tie → random among top scorers."""
    if not vote_counts:
        return None, False
    max_votes = max(vote_counts.values())
    top = [a for a, v in vote_counts.items() if v == max_votes]
    return random.choice(top), len(top) > 1


# ---------------------------------------------------------------------------
# PhaseContext — shared mutable state for one day's pending actions
# ---------------------------------------------------------------------------

class PhaseContext:
    """
    Holds all within-day pending data. Owned by PMEnvironment, reset each day.

    Attributes
    ----------
    pending_pre_task_msgs  : agent_id → PreTaskMessage
    pending_task_inputs    : agent_id → raw answer string
    pending_votes          : voter_id → target_id
    post_msg_counts        : day → sender_id → recipient_id → count
    day_questions          : agent_id → (question, correct_answer, options)
    """

    def __init__(self) -> None:
        self.pending_pre_task_msgs:  Dict[int, PreTaskMessage] = {}
        self.pending_task_inputs:    Dict[int, str]            = {}
        self.pending_votes:          Dict[int, int]            = {}
        self.post_msg_counts:        Dict[int, Dict[int, Dict[int, int]]] = {}
        self.day_questions:          Dict[int, Tuple[str, str, List[str]]] = {}
        # Trust-decision log — one entry per agent per task execution.
        # agent_id -> { trust_values, min_trust, max_trust, random_value,
        #               senders_with_answers, info_source, trust_of_source,
        #               used_shared_answer, solved_by_self }
        self.trust_decision_log: Dict[int, Dict] = {}

    def reset_day(self) -> None:
        self.pending_pre_task_msgs = {}
        self.pending_task_inputs   = {}
        self.pending_votes         = {}
        self.trust_decision_log    = {}

    def post_count(self, day: int, sender: int, recipient: int) -> int:
        return (
            self.post_msg_counts
            .get(day, {})
            .get(sender, {})
            .get(recipient, 0)
        )

    def increment_post_count(self, day: int, sender: int, recipient: int) -> None:
        d = self.post_msg_counts.setdefault(day, {})
        s = d.setdefault(sender, {})
        s[recipient] = s.get(recipient, 0) + 1


# ---------------------------------------------------------------------------
# Phase 1 — TASK_REVEAL
# ---------------------------------------------------------------------------

def enter_task_reveal(
    state: PMState,
    agents: Dict[int, Agent],
    ctx: PhaseContext,
    task_config: Dict,
) -> None:
    """
    Deal private info to each alive agent.
    Broadcast global public info.
    Immediately advance phase to PRE_DISCUSSION (no agent action needed).
    Mutates: state, agents, ctx.day_questions.
    """
    day       = state.day
    n_options = task_config["n_options"]

    ctx.day_questions = {}
    private_map: Dict[int, str] = {}

    # One shared question for all alive agents.
    shared_q, shared_correct, shared_options = _make_shared_question(
        n_options=n_options,
        day=day,
        n_agents=len(state.alive_agents),
    )

    # Assign each alive agent a different private candidate answer where possible.
    assigned_candidates = list(shared_options)
    random.shuffle(assigned_candidates)
    if len(assigned_candidates) < len(state.alive_agents):
        # Safety fallback; should rarely happen due to pool_size logic above.
        assigned_candidates.extend(
            [shared_correct] * (len(state.alive_agents) - len(assigned_candidates))
        )

    for idx, aid in enumerate(state.alive_agents):
        private_candidate = assigned_candidates[idx]
        ctx.day_questions[aid] = (shared_q, shared_correct, shared_options)
        private_info = (
            f"Task: {shared_q}\n"
            f"Your private candidate answer: {private_candidate}\n"
            f"All candidates: {', '.join(shared_options)}\n"
            f"(Exactly one candidate is correct. You may tell truth, twist, or lie.)"
        )

        private_map[aid]         = private_info
        agents[aid].private_info = private_info

    state.each_agent_private_info = private_map
    state.global_public_info = (
        f"Day {day} — Each agent holds a private answer. "
        f"Submit your best answer during Task Execution. "
        f"Points awarded for correct submissions."
    )

    # Phase 1 needs no agent action — transition immediately
    state.phase = Phase.PRE_DISCUSSION


# ---------------------------------------------------------------------------
# Phase 2 — PRE_DISCUSSION
# ---------------------------------------------------------------------------

def handle_pre_discussion(
    action: PMAction,
    state: PMState,
    agents: Dict[int, Agent],
    ctx: PhaseContext,
) -> None:
    """
    Validate and store the one pre-task message each agent sends.
    Raises ValueError on duplicate or wrong phase.
    """
    assert action.action_type == ActionType.SEND_PRE_TASK_MESSAGE, (
        f"Phase 2 expects SEND_PRE_TASK_MESSAGE, got {action.action_type}"
    )
    msg = action.pre_task_message
    assert msg is not None
    assert msg.sender_id == action.agent_id
    assert msg.day       == state.day

    if action.agent_id in ctx.pending_pre_task_msgs:
        raise ValueError(
            f"Agent {action.agent_id} already submitted a pre-task message on day {state.day}."
        )

    ctx.pending_pre_task_msgs[action.agent_id] = msg
    state.record_pre_task_message(msg)
    agents[action.agent_id].record_pre_task_message(msg)


def is_pre_discussion_complete(state: PMState, ctx: PhaseContext) -> bool:
    return all(aid in ctx.pending_pre_task_msgs for aid in state.alive_agents)


# ---------------------------------------------------------------------------
# Phase 3 — TASK_EXECUTION
# ---------------------------------------------------------------------------

def _extract_answer_from_message(content: str) -> Optional[str]:
    """
    Try to parse a numeric answer out of a pre-discussion message content.
    Returns the first token that looks like an integer, or None.
    """
    for token in content.split():
        cleaned = token.strip('.,!?;:\"\'')
        if cleaned.isdigit():
            return cleaned
    return None


def _trust_based_task_decision(
    agent_id: int,
    state: PMState,
    ctx: PhaseContext,
) -> Tuple[str, Dict]:
    """
    Core trust-decision logic for Phase 3.

    Steps
    -----
    1. Collect trust values this agent has for every other alive agent.
    2. Find min and max trust values.
    3. Generate a random float in [min_trust, max_trust].
    4. Find which senders shared an answer in Phase 2 that this agent can see.
    5. For the FIRST such sender whose trust_factor > random_value:
       → return their answer blindly (used_shared_answer = True)
    6. If no sender clears the threshold (or no answers were shared):
       → agent solves using own private info + all received messages
         from senders it did NOT blindly trust (solved_by_self = True)

    Returns
    -------
    (answer_string, decision_log_dict)
    """
    day   = state.day
    alive = [a for a in state.alive_agents if a != agent_id]

    # Step 1 — collect trust values
    trust_values: Dict[int, float] = {
        other: state.trust_scores_dict.get(agent_id, {}).get(other, 0.5)
        for other in alive
    }

    # Step 2 — min / max
    min_trust = min(trust_values.values()) if trust_values else 0.5
    max_trust = max(trust_values.values()) if trust_values else 0.5

    # Step 3 — random value in [min_trust, max_trust]
    rand_val = random.uniform(min_trust, max_trust)

    # Step 4 — find senders who shared an answer this agent can see
    visible_msgs = state.messages_visible_to(agent_id, day)
    senders_with_answers: List[int] = []
    sender_answers: Dict[int, str] = {}
    for msg in visible_msgs:
        if msg.sender_id == agent_id:
            continue
        parsed = _extract_answer_from_message(msg.content)
        if parsed is not None:
            senders_with_answers.append(msg.sender_id)
            sender_answers[msg.sender_id] = parsed

    # Step 5 — check each sender: if trust_of_sender > rand_val → use blindly
    info_source: Optional[int]   = None
    trust_of_source: Optional[float] = None
    used_shared_answer = False
    blind_answer: Optional[str] = None

    for sender_id in senders_with_answers:
        sender_trust = trust_values.get(sender_id, 0.5)
        if sender_trust > rand_val:
            # Trust exceeds random threshold → use this answer blindly
            info_source        = sender_id
            trust_of_source    = sender_trust
            used_shared_answer = True
            blind_answer       = sender_answers[sender_id]
            break   # first qualifying sender wins

    # Step 6 — decision
    if used_shared_answer and blind_answer is not None:
        final_answer  = blind_answer
        solved_by_self = False
    else:
        # Solve using own private info + untrusted received messages
        _, correct, _ = ctx.day_questions.get(agent_id, (None, "", None))
        # In a real system, the agent LLM would reason here.
        # Stub: use own correct answer (represents the agent reasoning correctly).
        final_answer   = correct
        solved_by_self = True

    decision_log = {
        "trust_values":         trust_values,
        "min_trust":            round(min_trust, 4),
        "max_trust":            round(max_trust, 4),
        "random_value":         round(rand_val, 4),
        "senders_with_answers": senders_with_answers,
        "info_source":          info_source,
        "trust_of_source":      round(trust_of_source, 4) if trust_of_source is not None else None,
        "used_shared_answer":   used_shared_answer,
        "solved_by_self":       solved_by_self,
    }

    return final_answer, decision_log


def handle_task_execution(
    action: PMAction,
    state: PMState,
    ctx: PhaseContext,
) -> None:
    """
    Store the task answer submitted by an agent.
    If the agent has not yet been processed by the trust-decision logic,
    that is handled inside finalise_task_execution (bulk processing).
    Direct submissions (e.g. from an LLM orchestrator) are accepted as-is.
    """
    assert action.action_type == ActionType.SUBMIT_TASK_INPUT, (
        f"Phase 3 expects SUBMIT_TASK_INPUT, got {action.action_type}"
    )
    assert action.task_input is not None

    if action.agent_id not in ctx.pending_task_inputs:
        ctx.pending_task_inputs[action.agent_id] = action.task_input


def is_task_execution_complete(state: PMState, ctx: PhaseContext) -> bool:
    return all(aid in ctx.pending_task_inputs for aid in state.alive_agents)


def finalise_task_execution(
    state: PMState,
    agents: Dict[int, Agent],
    ctx: PhaseContext,
    task_config: Dict,
) -> Dict[int, float]:
    """
    For each alive agent:
      1. Run trust-based decision logic to determine answer source.
      2. Store decision in ctx.trust_decision_log[agent_id].
      3. Score the answer, apply lie/twist penalties.
      4. Emit TaskResult.

    Returns partial reward dict (task component, w1=0.4).
    """
    day         = state.day
    correct_pts = task_config["correct_answer_points"]
    lie_penalty = task_config["lie_penalty"]
    lie_ack_pen = task_config["lie_acknowledged_penalty"]

    per_agent_outcome: Dict[int, int] = {}
    ground_truths:     Dict[int, str] = {}

    for aid in state.alive_agents:
        _, correct, _ = ctx.day_questions[aid]
        ground_truths[aid] = correct

        # ── Trust-based answer decision ──────────────────────────────
        final_answer, decision_log = _trust_based_task_decision(aid, state, ctx)
        ctx.trust_decision_log[aid] = decision_log

        # Override / record the answer chosen by the trust logic
        ctx.pending_task_inputs[aid] = final_answer

        # ── Scoring ──────────────────────────────────────────────────
        pts = correct_pts if final_answer.strip() == correct.strip() else 0

        msg = state.pre_task_messages.get(day, {}).get(aid)
        if msg is not None:
            if msg.veracity == MessageVeracity.LIE:
                pts -= lie_penalty
            elif msg.veracity == MessageVeracity.TWIST:
                pts -= lie_ack_pen

        per_agent_outcome[aid]      = max(0, pts)
        state.agents_point_map[aid] = state.agents_point_map.get(aid, 0) + per_agent_outcome[aid]
        agents[aid].points         += per_agent_outcome[aid]

    ground_truth_str = "; ".join(f"agent_{k}={v}" for k, v in ground_truths.items())

    state.task_results[day] = TaskResult(
        day=day,
        task_type=state.task_type,
        per_agent_outcome=per_agent_outcome,
        ground_truth_exposed=ground_truth_str,
    )

    return {
        aid: float(per_agent_outcome.get(aid, 0)) * 0.4
        for aid in state.alive_agents
    }


# ---------------------------------------------------------------------------
# Phase 4 — POST_DISCUSSION
# ---------------------------------------------------------------------------

def handle_post_discussion(
    action: PMAction,
    state: PMState,
    agents: Dict[int, Agent],
    ctx: PhaseContext,
    task_config: Dict,
) -> None:
    """
    Accept PostDiscussionMessage or TrustAssessment.
    Enforces per-pair message cap defined in task_config.
    """
    day = state.day
    cap = task_config["max_post_discussion_messages"]

    if action.action_type == ActionType.SEND_POST_DISCUSSION_MSG:
        msg = action.post_discussion_msg
        assert msg is not None
        assert msg.sender_id    == action.agent_id
        assert msg.recipient_id != action.agent_id
        assert msg.recipient_id in state.alive_agents

        current = ctx.post_count(day, msg.sender_id, msg.recipient_id)
        if current >= cap:
            raise ValueError(
                f"Agent {msg.sender_id} has hit the message cap ({cap}) "
                f"with agent {msg.recipient_id} today."
            )

        state.post_discussion_messages.setdefault(day, []).append(msg)
        ctx.increment_post_count(day, msg.sender_id, msg.recipient_id)

    elif action.action_type == ActionType.SUBMIT_TRUST_ASSESSMENT:
        assessment = action.trust_assessment
        assert assessment is not None
        assert assessment.assessor_id == action.agent_id

        state.apply_trust_assessment(assessment)
        agents[action.agent_id].update_trust(
            assessment.target_id, assessment.delta.to_float()
        )

    else:
        raise ValueError(
            f"Phase 4 expects SEND_POST_DISCUSSION_MSG or SUBMIT_TRUST_ASSESSMENT, "
            f"got {action.action_type}"
        )


def is_post_discussion_complete(state: PMState) -> bool:
    """
    Advance once every alive agent has submitted a TrustAssessment
    for every other alive agent (n × (n-1) total).
    """
    expected     = len(state.alive_agents) * (len(state.alive_agents) - 1)
    day_assessments = state.trust_assessments.get(state.day, [])
    return len(day_assessments) >= expected


# ---------------------------------------------------------------------------
# Phase 5 — VOTING
# ---------------------------------------------------------------------------

def handle_voting(
    action: PMAction,
    state: PMState,
    ctx: PhaseContext,
) -> None:
    """Collect one vote (with reason stored in info dict) per alive agent."""
    assert action.action_type == ActionType.VOTE, (
        f"Phase 5 expects VOTE, got {action.action_type}"
    )
    assert action.vote_target is not None
    assert action.vote_target != action.agent_id,  "Cannot vote for yourself."
    assert action.vote_target in state.alive_agents, "Target is not alive."

    if action.agent_id not in ctx.pending_votes:
        ctx.pending_votes[action.agent_id] = action.vote_target


def is_voting_complete(state: PMState, ctx: PhaseContext) -> bool:
    return all(aid in ctx.pending_votes for aid in state.alive_agents)


def finalise_voting(
    state: PMState,
    agents: Dict[int, Agent],
    ctx: PhaseContext,
    vote_reasons: Dict[int, str],       # voter_id → reason string
) -> Tuple[Dict[int, float], Optional[int]]:
    """
    Tally votes, emit VoteRecord, eliminate the loser.
    Returns (rewards_dict, eliminated_id).
    Mutates: state.alive_agents, state.vote_history, agents[*].alive.
    """
    alive = list(state.alive_agents)

    vote_counts: Dict[int, int] = {aid: 0 for aid in alive}
    for _voter, target in ctx.pending_votes.items():
        vote_counts[target] = vote_counts.get(target, 0) + 1

    eliminated_id, was_tie = _resolve_tie(vote_counts, alive)

    state.vote_history.append(VoteRecord(
        day=state.day,
        votes_cast=dict(ctx.pending_votes),
        vote_counts=vote_counts,
        eliminated_id=eliminated_id,
        was_tie=was_tie,
    ))

    rewards: Dict[int, float] = {}
    for aid in alive:
        r  = 0.0
        r += 1.0 if aid != eliminated_id else 0.0          # survival  w2=1.0
        if ctx.pending_votes.get(aid) == eliminated_id:
            r += 0.3                                        # influence w3=0.3
        rewards[aid] = r

    if eliminated_id is not None:
        state.alive_agents = [a for a in alive if a != eliminated_id]
        state.agent_removed_dict[eliminated_id] = state.day
        agents[eliminated_id].alive = False

    return rewards, eliminated_id


# ---------------------------------------------------------------------------
# Phase-entry helper for Phase 5 — build public reveal
# ---------------------------------------------------------------------------

def build_public_reveal(state: PMState) -> DayPublicReveal:
    """
    Aggregate honesty stats from Phase 2 messages.
    TWIST is counted as an acknowledged lie.
    """
    day   = state.day
    alive = state.alive_agents

    lies_told:           Dict[int, int] = {aid: 0 for aid in alive}
    lies_acknowledged:   Dict[int, int] = {aid: 0 for aid in alive}
    lies_unacknowledged: Dict[int, int] = {aid: 0 for aid in alive}
    task_scores:         Dict[int, int] = {
        aid: state.agents_point_map.get(aid, 0) for aid in alive
    }

    for aid in alive:
        msg = state.pre_task_messages.get(day, {}).get(aid)
        if msg is None:
            continue
        if msg.veracity == MessageVeracity.LIE:
            lies_told[aid]            = 1
            lies_unacknowledged[aid]  = 1
        elif msg.veracity == MessageVeracity.TWIST:
            lies_told[aid]            = 1
            lies_acknowledged[aid]    = 1
            lies_unacknowledged[aid]  = 0

    return DayPublicReveal(
        day=day,
        lies_told=lies_told,
        lies_acknowledged=lies_acknowledged,
        lies_unacknowledged=lies_unacknowledged,
        task_scores=task_scores,
    )