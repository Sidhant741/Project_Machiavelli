"""
compression.py — Project Machiavelli
======================================
Day-level structured summary/compression for each alive agent.

Storage format (per agent, per day):
  {
    "day": int,
    "agent_id": int,

    "pre_discussion": {
      "protected_info":   str,          # what this agent kept hidden
      "private_info":     str,          # own private info for the day
      "other_agents":     {             # what each other agent shared with this agent
        agent_id: {
          "message":   str,
          "veracity":  "truth"|"twist"|"lie",   # revealed at end of day
        }
      },
      "own_message": {
        "content":  str,
        "veracity": "truth"|"twist"|"lie",
        "recipient": int | None,        # None = broadcast
      },
    },

    "task": {
      "own_answer":      str,
      "correct_answer":  str,
      "score":           int,
      "ground_truth":    str,           # full ground truth string
    },

    "post_discussion": {
      "lies_told":            int,
      "truths_told":          int,      # messages with veracity==TRUTH this day
      "lies_acknowledged":    int,
      "latest_trust_factors": {         # trust values AFTER Phase 4 updates
        agent_id: float,
      },
      "messages_exchanged": [           # all Phase 4 messages this agent sent/received
        {
          "direction":    "sent"|"received",
          "other_agent":  int,
          "content":      str,
          "turn_index":   int,
        }
      ],
    },

    "voting": {
      "own_vote": {
        "voted_for": int,
        "reason":    str,
      },
      "votes_received":  int,           # how many agents voted for this agent
      "vote_counts":     { agent_id: int },
      "eliminated":      int | None,
      "survived":        bool,
    },
  }

Only alive agents at END of the day get a summary entry.
(Eliminated agent's summary is skipped — they are gone.)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from ..models import (
        Agent, PMState, Phase,
        MessageVeracity, VoteRecord, TaskResult, DayPublicReveal,
        PostDiscussionMessage,
        AgentPriorSnapshot, EpisodeRecord,
    )
except ImportError:  # running from project root
    from models import (
        Agent, PMState, Phase,
        MessageVeracity, VoteRecord, TaskResult, DayPublicReveal,
        PostDiscussionMessage,
        AgentPriorSnapshot, EpisodeRecord,
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def compress_day(
    day: int,
    state: PMState,
    agents: Dict[int, Agent],
    vote_reasons: Dict[int, str],          # voter_id -> reason string
    protected_info_map: Dict[int, str],    # agent_id -> what they kept hidden
    trust_decision_log: Dict[int, Dict],   # agent_id -> trust decision record
) -> Dict[int, Dict[str, Any]]:
    """
    Build a structured key-value summary for every ALIVE agent at end of day.
    Eliminated agent is excluded (they are already removed from state.alive_agents).

    Parameters
    ----------
    day               : day number just completed
    state             : full PMState (post-elimination)
    agents            : dict of Agent objects
    vote_reasons      : { voter_id: reason_string } collected during voting phase
    protected_info_map: { agent_id: protected_info_string } set during task_reveal

    Returns
    -------
    { agent_id: summary_dict }  — one entry per alive agent
    """
    summaries: Dict[int, Dict[str, Any]] = {}

    for agent_id in state.alive_agents:
        summaries[agent_id] = _build_agent_summary(
            day=day,
            agent_id=agent_id,
            state=state,
            agents=agents,
            vote_reasons=vote_reasons,
            protected_info_map=protected_info_map,
            trust_decision_log=trust_decision_log,
        )

    return summaries


# ---------------------------------------------------------------------------
# Per-agent summary builder
# ---------------------------------------------------------------------------

def _build_agent_summary(
    day: int,
    agent_id: int,
    state: PMState,
    agents: Dict[int, Agent],
    vote_reasons: Dict[int, str],
    protected_info_map: Dict[int, str],
    trust_decision_log: Dict[int, Dict],
) -> Dict[str, Any]:

    return {
        "day":      day,
        "agent_id": agent_id,

        "pre_discussion": _summarise_pre_discussion(
            day, agent_id, state, protected_info_map
        ),
        "task": _summarise_task(
            day, agent_id, state
        ),
        "task_decision": _summarise_task_decision(
            agent_id, trust_decision_log
        ),
        "post_discussion": _summarise_post_discussion(
            day, agent_id, state, agents
        ),
        "voting": _summarise_voting(
            day, agent_id, state, vote_reasons
        ),
    }


# ---------------------------------------------------------------------------
# Phase-level summarisers
# ---------------------------------------------------------------------------

def _summarise_pre_discussion(
    day: int,
    agent_id: int,
    state: PMState,
    protected_info_map: Dict[int, str],
) -> Dict[str, Any]:
    """
    pre_discussion section:
      - protected_info   : what this agent kept hidden (never shared)
      - private_info     : own private info for the day
      - own_message      : what this agent sent, veracity, recipient
      - other_agents     : messages this agent received + revealed veracity
    """
    day_msgs = state.pre_task_messages.get(day, {})

    # Own message
    own_msg = day_msgs.get(agent_id)
    own_message: Dict[str, Any] = {}
    if own_msg is not None:
        own_message = {
            "content":   own_msg.content,
            "veracity":  own_msg.veracity.value,
            "recipient": own_msg.recipient_id,
        }

    # Messages received from others (visible = broadcast or addressed to this agent)
    other_agents: Dict[int, Dict[str, Any]] = {}
    for sender_id, msg in day_msgs.items():
        if sender_id == agent_id:
            continue
        # Visibility: broadcast (recipient_id None) or explicitly sent to this agent
        if msg.recipient_id is None or msg.recipient_id == agent_id:
            other_agents[sender_id] = {
                "message":  msg.content,
                "veracity": msg.veracity.value,   # revealed at end-of-day
            }

    return {
        "protected_info":  protected_info_map.get(agent_id, ""),
        "private_info":    state.each_agent_private_info.get(agent_id, ""),
        "own_message":     own_message,
        "other_agents":    other_agents,
    }


def _summarise_task(
    day: int,
    agent_id: int,
    state: PMState,
) -> Dict[str, Any]:
    """
    task section:
      - own_answer     : what this agent submitted
      - correct_answer : the right answer (from ground truth)
      - score          : points earned this round
      - ground_truth   : full ground truth string broadcast at Phase 3 end
    """
    result: Optional[TaskResult] = state.task_results.get(day)

    if result is None:
        return {
            "own_answer":     "",
            "correct_answer": "",
            "score":          0,
            "ground_truth":   "",
        }

    # Parse correct answer for this agent out of ground_truth_exposed string
    # Format: "agent_0=42; agent_1=17; ..."
    correct_answer = ""
    if result.ground_truth_exposed:
        for part in result.ground_truth_exposed.split(";"):
            part = part.strip()
            if f"agent_{agent_id}=" in part:
                correct_answer = part.split("=", 1)[-1].strip()
                break

    return {
        "own_answer":     "",          # filled by env from pending_task_inputs snapshot
        "correct_answer": correct_answer,
        "score":          result.per_agent_outcome.get(agent_id, 0),
        "ground_truth":   result.ground_truth_exposed or "",
    }


def _summarise_task_decision(
    agent_id: int,
    trust_decision_log: Dict[int, Dict],
) -> Dict[str, Any]:
    """
    task_decision section — records HOW the agent chose its answer.

    Fields
    ------
    trust_values        : { sender_id: float }  trust scores for all alive peers
    min_trust           : float                  minimum trust seen
    max_trust           : float                  maximum trust seen
    random_value        : float                  drawn uniformly from [min, max]
    senders_with_answers: [int]                  senders who shared a parseable answer
    info_source         : int | None             sender whose answer was used blindly
    trust_of_source     : float | None           trust for that sender
    used_shared_answer  : bool  True  -> random_value < trust_of_source
                                       -> answer taken from sender, no reasoning
    solved_by_self      : bool  True  -> random_value >= trust_of_source (or no source)
                                       -> agent reasoned from own private info
    """
    log = trust_decision_log.get(agent_id)
    if log is None:
        return {
            "trust_values":          {},
            "min_trust":             None,
            "max_trust":             None,
            "random_value":          None,
            "senders_with_answers":  [],
            "info_source":           None,
            "trust_of_source":       None,
            "used_shared_answer":    False,
            "solved_by_self":        True,
        }
    return {
        "trust_values":          log.get("trust_values", {}),
        "min_trust":             log.get("min_trust"),
        "max_trust":             log.get("max_trust"),
        "random_value":          log.get("random_value"),
        "senders_with_answers":  log.get("senders_with_answers", []),
        "info_source":           log.get("info_source"),
        "trust_of_source":       log.get("trust_of_source"),
        "used_shared_answer":    log.get("used_shared_answer", False),
        "solved_by_self":        log.get("solved_by_self", True),
    }


def _summarise_post_discussion(
    day: int,
    agent_id: int,
    state: PMState,
    agents: Dict[int, Agent],
) -> Dict[str, Any]:
    """
    post_discussion section:
      - lies_told            : number of LIE messages this agent sent today
      - truths_told          : number of TRUTH messages this agent sent today
      - lies_acknowledged    : from DayPublicReveal
      - latest_trust_factors : trust values AFTER Phase 4 updates
      - messages_exchanged   : all Phase 4 messages sent/received by this agent
    """
    # Count lies / truths from Phase 2 message (only 1 per agent per day)
    day_msgs   = state.pre_task_messages.get(day, {})
    own_msg    = day_msgs.get(agent_id)
    lies_told  = 0
    truths_told = 0
    if own_msg is not None:
        if own_msg.veracity == MessageVeracity.LIE:
            lies_told = 1
        elif own_msg.veracity == MessageVeracity.TRUTH:
            truths_told = 1

    # Lies acknowledged (from public reveal if available)
    reveal: Optional[DayPublicReveal] = state.public_reveals.get(day)
    lies_acknowledged = (
        reveal.lies_acknowledged.get(agent_id, 0) if reveal else 0
    )

    # Latest trust factors (post Phase 4 updates)
    latest_trust: Dict[int, float] = dict(
        state.trust_scores_dict.get(agent_id, {})
    )

    # Phase 4 messages this agent was part of
    all_post_msgs: List[PostDiscussionMessage] = state.post_discussion_messages.get(day, [])
    messages_exchanged: List[Dict[str, Any]] = []
    for m in all_post_msgs:
        if m.sender_id == agent_id:
            messages_exchanged.append({
                "direction":   "sent",
                "other_agent": m.recipient_id,
                "content":     m.content,
                "turn_index":  m.turn_index,
            })
        elif m.recipient_id == agent_id:
            messages_exchanged.append({
                "direction":   "received",
                "other_agent": m.sender_id,
                "content":     m.content,
                "turn_index":  m.turn_index,
            })

    return {
        "lies_told":            lies_told,
        "truths_told":          truths_told,
        "lies_acknowledged":    lies_acknowledged,
        "latest_trust_factors": latest_trust,
        "messages_exchanged":   messages_exchanged,
    }


def _summarise_voting(
    day: int,
    agent_id: int,
    state: PMState,
    vote_reasons: Dict[int, str],
) -> Dict[str, Any]:
    """
    voting section:
      - own_vote        : { voted_for, reason }
      - votes_received  : how many agents voted for this agent
      - vote_counts     : full tally { agent_id: count }
      - eliminated      : agent_id who was eliminated (None if tie unresolved)
      - survived        : bool
    """
    vote_record: Optional[VoteRecord] = (
        state.vote_history[-1] if state.vote_history else None
    )

    if vote_record is None or vote_record.day != day:
        return {
            "own_vote":       {"voted_for": None, "reason": ""},
            "votes_received": 0,
            "vote_counts":    {},
            "eliminated":     None,
            "survived":       agent_id in state.alive_agents,
        }

    voted_for = vote_record.votes_cast.get(agent_id)

    return {
        "own_vote": {
            "voted_for": voted_for,
            "reason":    vote_reasons.get(agent_id, ""),
        },
        "votes_received": vote_record.vote_counts.get(agent_id, 0),
        "vote_counts":    dict(vote_record.vote_counts),
        "eliminated":     vote_record.eliminated_id,
        "survived":       agent_id in state.alive_agents,
    }


# ---------------------------------------------------------------------------
# Storage helper — attach summaries to agents and env-level store
# ---------------------------------------------------------------------------

def store_day_summaries(
    day_summaries: Dict[int, Dict[str, Any]],
    agents: Dict[int, Agent],
    global_store: Dict[str, Any],   # env-level dict, mutated in-place
    task_inputs_snapshot: Dict[int, str],  # pending_task_inputs before clearing
) -> None:
    """
    1. Patch own_answer into each summary from the task_inputs snapshot.
    2. Store under agents[id].day_history-compatible key for fast lookup.
    3. Write into global_store["day_{N}"][agent_id] = summary.

    Parameters
    ----------
    day_summaries         : output of compress_day()
    agents                : Agent objects to attach summaries to
    global_store          : env.summary_store dict (mutated)
    task_inputs_snapshot  : copy of ctx.pending_task_inputs taken before clearing
    """
    if not day_summaries:
        return

    day = next(iter(day_summaries.values()))["day"]
    day_key = f"day_{day}"
    global_store.setdefault(day_key, {})

    for agent_id, summary in day_summaries.items():
        # Patch own_answer (was left blank during build to avoid dependency order)
        summary["task"]["own_answer"] = task_inputs_snapshot.get(agent_id, "")

        # Write to global store  →  global_store["day_3"][0] = {...}
        global_store[day_key][agent_id] = summary

    # ── Pretty-print helper (for debugging) ──────────────────────────────
    _log_summary_store(day_key, global_store[day_key])


# ---------------------------------------------------------------------------
# Debug logger
# ---------------------------------------------------------------------------

def _log_summary_store(day_key: str, day_store: Dict[int, Dict[str, Any]]) -> None:
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  COMPRESSION STORE — {day_key}")
    print(sep)
    for agent_id, summary in day_store.items():
        print(f"\n  ── Agent {agent_id} ──")
        _print_kv("pre_discussion.protected_info",
                  summary["pre_discussion"]["protected_info"][:80])
        _print_kv("pre_discussion.own_message.veracity",
                  summary["pre_discussion"].get("own_message", {}).get("veracity", "—"))
        _print_kv("pre_discussion.other_agents",
                  {k: v["veracity"] for k, v in summary["pre_discussion"]["other_agents"].items()})
        _print_kv("task.score",          summary["task"]["score"])
        _print_kv("task.correct_answer", summary["task"]["correct_answer"])
        td = summary.get("task_decision", {})
        _print_kv("task_decision.trust_values",        td.get("trust_values", {}))
        _print_kv("task_decision.min_trust",           td.get("min_trust"))
        _print_kv("task_decision.max_trust",           td.get("max_trust"))
        _print_kv("task_decision.random_value",        td.get("random_value"))
        _print_kv("task_decision.senders_with_answers",td.get("senders_with_answers", []))
        _print_kv("task_decision.info_source",         td.get("info_source"))
        _print_kv("task_decision.trust_of_source",     td.get("trust_of_source"))
        _print_kv("task_decision.used_shared_answer",  td.get("used_shared_answer"))
        _print_kv("task_decision.solved_by_self",      td.get("solved_by_self"))
        _print_kv("post_discussion.lies_told",         summary["post_discussion"]["lies_told"])
        _print_kv("post_discussion.truths_told",        summary["post_discussion"]["truths_told"])
        _print_kv("post_discussion.lies_acknowledged",  summary["post_discussion"]["lies_acknowledged"])
        _print_kv("post_discussion.latest_trust_factors",
                  summary["post_discussion"]["latest_trust_factors"])
        _print_kv("voting.own_vote",       summary["voting"]["own_vote"])
        _print_kv("voting.votes_received", summary["voting"]["votes_received"])
        _print_kv("voting.eliminated",     summary["voting"]["eliminated"])
        _print_kv("voting.survived",       summary["voting"]["survived"])
    print(f"\n{sep}\n")


def _print_kv(key: str, value: Any) -> None:
    print(f"    {key:<45} : {value}")


# ---------------------------------------------------------------------------
# Episode-level compression
# ---------------------------------------------------------------------------

def compress_episode(
    episode_index: int,
    task: str,
    state: PMState,
    agents: Dict[int, Agent],
    summary_store: Dict[str, Any],
) -> EpisodeRecord:
    """
    Build an EpisodeRecord at the end of a completed episode.

    Collects, per agent:
      - Every day summary dict from summary_store where the agent appears
        (i.e. days they were alive), stored as a list ordered by day.
      - A prior snapshot: truthful_prior, deception_prior, risk_beta,
        and final trust_scores from the Agent object.

    Also computes:
      - winner_ids      : agents still alive at game end
      - eliminated_order: agents sorted by the day they were removed

    Parameters
    ----------
    episode_index : 0-based counter managed by PMEnvironment
    task          : difficulty string ("easy" | "medium" | "hard")
    state         : full PMState at game end
    agents        : dict of all Agent objects (alive + eliminated)
    summary_store : env.summary_store — { "day_N": { agent_id: summary_dict } }

    Returns
    -------
    EpisodeRecord
    """
    all_agent_ids = list(agents.keys())
    winner_ids    = list(state.alive_agents)

    # Eliminated order: sort by removal day (ascending  = first eliminated first)
    eliminated_order: List[int] = sorted(
        state.agent_removed_dict.keys(),
        key=lambda aid: state.agent_removed_dict[aid],
    )

    # ── Per-agent day summaries (only days agent was alive / has entry) ──
    agent_day_summaries: Dict[int, List[Dict]] = {aid: [] for aid in all_agent_ids}

    # Collect day keys sorted numerically (day_1, day_2, ...)
    day_keys_sorted = sorted(
        summary_store.keys(),
        key=lambda k: int(k.split("_")[-1]),
    )
    for day_key in day_keys_sorted:
        day_store = summary_store[day_key]        # { agent_id: summary_dict }
        for aid in all_agent_ids:
            if aid in day_store:
                agent_day_summaries[aid].append(dict(day_store[aid]))

    # ── Prior snapshots ─────────────────────────────────────────────────
    prior_snapshots: Dict[int, AgentPriorSnapshot] = {}
    for aid, agent in agents.items():
        prior_snapshots[aid] = AgentPriorSnapshot(
            agent_id=aid,
            episode_index=episode_index,
            truthful_prior=agent.truthful_prior,
            deception_prior=agent.deception_prior,
            risk_beta=agent.risk_beta,
            final_trust_scores=dict(state.trust_scores_dict.get(aid, {})),
        )

    record = EpisodeRecord(
        episode_index=episode_index,
        task=task,
        n_agents=len(all_agent_ids),
        days_played=state.day,
        winner_ids=winner_ids,
        eliminated_order=eliminated_order,
        agent_day_summaries=agent_day_summaries,
        prior_snapshots=prior_snapshots,
    )

    _log_episode_record(record)
    return record


def _log_episode_record(record: EpisodeRecord) -> None:
    """Pretty-print the episode-level record for debugging."""
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  EPISODE COMPRESSION — episode {record.episode_index}")
    print(sep)
    print(f"    task              : {record.task}")
    print(f"    n_agents          : {record.n_agents}")
    print(f"    days_played       : {record.days_played}")
    print(f"    winner_ids        : {record.winner_ids}")
    print(f"    eliminated_order  : {record.eliminated_order}")
    for aid, day_summaries in record.agent_day_summaries.items():
        snap = record.prior_snapshots.get(aid)
        print(f"\n  ── Agent {aid} ──")
        print(f"    days_survived     : {len(day_summaries)}")
        if snap:
            print(f"    truthful_prior    : {snap.truthful_prior}")
            print(f"    deception_prior   : {snap.deception_prior}")
            print(f"    risk_beta         : {snap.risk_beta}")
            print(f"    final_trust_scores: {snap.final_trust_scores}")
        for ds in day_summaries:
            print(f"    [day {ds['day']}] score={ds['task']['score']}, "
                  f"survived={ds['voting']['survived']}")
    print(f"\n{sep}\n")