"""
PM Environment — Project Machiavelli
Complete implementation of PMEnvironment based on models.py, config.py and README.

Phase flow per day:
  Phase 1 — TASK_REVEAL:      public info broadcast, private info dealt per agent
  Phase 2 — PRE_DISCUSSION:   each agent sends 1 message (truth / twist / lie)
  Phase 3 — TASK_EXECUTION:   task performed, results published
  Phase 4 — POST_DISCUSSION:  agents talk (max N messages per pair), trust updates
  Phase 5 — VOTING:           silent simultaneous vote, one agent eliminated
"""

from __future__ import annotations

import random
import json
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Relative / absolute import shim (works both as a package and standalone)
# ---------------------------------------------------------------------------
try:
    from ..models import (
        Agent, PMState, PMObservation, PMAction,
        ActionType, Phase, TaskType, TaskResult,
        PreTaskMessage, PostDiscussionMessage,
        TrustAssessment, TrustDelta, MessageVeracity,
        DayPublicReveal, VoteRecord, DayHistoryEntry,
    )
    from .config import GAME_CONFIGS
    from .utils import (
        generate_trivia_question,
        llm_call,
        build_day_summary_prompt,
    )
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    from models import (
        Agent, PMState, PMObservation, PMAction,
        ActionType, Phase, TaskType, TaskResult,
        PreTaskMessage, PostDiscussionMessage,
        TrustAssessment, TrustDelta, MessageVeracity,
        DayPublicReveal, VoteRecord, DayHistoryEntry,
    )
    from server.config import GAME_CONFIGS
    from server.utils import (
        generate_trivia_question,
        llm_call,
        build_day_summary_prompt,
    )


# ---------------------------------------------------------------------------
# Helpers — question / private-info generation (self-contained fallback)
# ---------------------------------------------------------------------------

def _make_question(n_options: int, day: int, agent_id: int) -> Tuple[str, str, List[str]]:
    """
    Returns (question_text, correct_answer, all_options_list).
    In a real deployment, replace with a proper trivia/DB lookup.
    """
    correct = str(random.randint(1, 100))
    distractors = []
    while len(distractors) < n_options - 1:
        d = str(random.randint(1, 100))
        if d != correct and d not in distractors:
            distractors.append(d)
    options = [correct] + distractors
    random.shuffle(options)
    question = (
        f"Day {day} task for agent {agent_id}: "
        f"Guess the secret number. Options: {', '.join(options)}"
    )
    return question, correct, options


def _resolve_tie(vote_counts: Dict[int, int], alive: List[int]) -> Tuple[Optional[int], bool]:
    """Return (eliminated_id, was_tie). On tie → random choice among tied agents."""
    if not vote_counts:
        return None, False
    max_votes = max(vote_counts.values())
    top = [a for a, v in vote_counts.items() if v == max_votes]
    was_tie = len(top) > 1
    return random.choice(top), was_tie


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class PMEnvironment:
    """
    Project Machiavelli — multi-agent social survival environment.

    Usage
    -----
    env = PMEnvironment()
    obs_map = env.reset(task="easy")          # {agent_id: PMObservation}

    # Inside your agent loop:
    action = PMAction(agent_id=0, action_type=ActionType.SEND_PRE_TASK_MESSAGE, ...)
    obs_map, rewards, done, info = env.step(action)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self.state: Optional[PMState]           = None
        self.agents: Dict[int, Agent]           = {}
        self.task_config: Dict[str, Any]        = {}
        self.task:  str                         = "easy"
        self.is_done: bool                      = False

        # Pending collections — filled during a phase, consumed at transition
        self._pending_pre_task_msgs: Dict[int, PreTaskMessage]  = {}
        self._pending_task_inputs:   Dict[int, str]             = {}
        self._pending_votes:         Dict[int, int]             = {}
        # day → {sender → {recipient → count}} for Phase 4 cap
        self._post_msg_counts: Dict[int, Dict[int, Dict[int, int]]] = {}

        # Per-day question bank: agent_id → (question, correct_answer, options)
        self._day_questions: Dict[int, Tuple[str, str, List[str]]] = {}

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, task: Optional[str] = None) -> Dict[int, PMObservation]:
        """
        Initialise a fresh episode.

        Parameters
        ----------
        task : "easy" | "medium" | "hard" (or prefixed "task_easy" etc.)
               If None, chosen at random.

        Returns
        -------
        obs_map : {agent_id: PMObservation} for Phase 1
        """
        if task is None:
            self.task = random.choice(["easy", "medium", "hard"])
        else:
            normalised = task.replace("task_", "")
            assert normalised in GAME_CONFIGS, (
                f"task '{task}' must be easy | medium | hard"
            )
            self.task = normalised

        cfg = GAME_CONFIGS[self.task]
        self.task_config = cfg

        n_agents: int      = cfg["n_agents"]          # always 4
        task_type_str: str = cfg.get("task_type", "individual")
        task_type = TaskType(task_type_str) if task_type_str != "both" else TaskType.INDIVIDUAL

        agent_ids = list(range(n_agents))

        # Build Agent objects
        self.agents = {
            aid: Agent(
                id=aid,
                trust_scores={other: 0.5 for other in agent_ids if other != aid},
            )
            for aid in agent_ids
        }

        # Build environment state
        self.state = PMState(
            day=1,
            phase=Phase.TASK_REVEAL,
            alive_agents=agent_ids,
            task_type=task_type,
            task_rules=self._describe_task_rules(),
            trust_scores_dict={
                aid: {other: 0.5 for other in agent_ids if other != aid}
                for aid in agent_ids
            },
            agents_point_map={aid: 0 for aid in agent_ids},
        )

        self.is_done = False
        self._pending_pre_task_msgs = {}
        self._pending_task_inputs   = {}
        self._pending_votes         = {}
        self._post_msg_counts       = {}

        # Phase 1: deal private info and broadcast public info
        self._enter_phase_task_reveal()

        return self._obs_map()

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(
        self, action: PMAction
    ) -> Tuple[Dict[int, PMObservation], Dict[int, float], bool, Dict]:
        """
        Accept one action from one agent. Returns updated observations for
        all alive agents, per-agent rewards (non-zero only at day end),
        done flag, and an info dict.

        The environment automatically advances the phase once all required
        actions for the current phase have been collected.
        """
        assert self.state is not None, "Call reset() first."
        assert not self.is_done, "Episode is finished."
        assert action.agent_id in self.state.alive_agents, (
            f"Agent {action.agent_id} is not alive."
        )

        phase = self.state.phase

        if phase == Phase.TASK_REVEAL:
            # Phase 1 has no agent actions — env transitions automatically.
            # Guard against stale calls.
            pass

        elif phase == Phase.PRE_DISCUSSION:
            self._handle_pre_discussion(action)

        elif phase == Phase.TASK_EXECUTION:
            self._handle_task_execution(action)

        elif phase == Phase.POST_DISCUSSION:
            self._handle_post_discussion(action)

        elif phase == Phase.VOTING:
            self._handle_voting(action)

        # Try to advance the phase
        rewards = self._try_advance_phase()

        obs = self._obs_map()
        info = {
            "day":   self.state.day,
            "phase": self.state.phase,
            "alive": self.state.alive_agents,
        }
        return obs, rewards, self.is_done, info

    # ------------------------------------------------------------------
    # get_observation()
    # ------------------------------------------------------------------

    def get_observation(
        self, agent_id: int, reveal_veracity: bool = False
    ) -> PMObservation:
        assert self.state is not None
        return PMObservation.from_state(
            self.state, agent_id, reveal_veracity=reveal_veracity
        )

    # ------------------------------------------------------------------
    # Phase entry logic
    # ------------------------------------------------------------------

    def _enter_phase_task_reveal(self) -> None:
        """Phase 1: generate questions, assign private info, broadcast public info."""
        assert self.state is not None
        day = self.state.day
        cfg = self.task_config
        n_options: int = cfg["n_options"]

        self._day_questions = {}
        private_map: Dict[int, str] = {}

        for aid in self.state.alive_agents:
            question, correct, options = _make_question(n_options, day, aid)
            self._day_questions[aid] = (question, correct, options)

            if n_options == 1:
                private_info = f"Your answer for today's task is: {correct}"
            else:
                private_info = (
                    f"Task: {question}\n"
                    f"Your options: {', '.join(options)}\n"
                    f"(One of these is correct.)"
                )

            private_map[aid] = private_info
            self.agents[aid].private_info = private_info

        self.state.each_agent_private_info = private_map
        self.state.global_public_info = (
            f"Day {day} — Task: Agents each hold a private answer. "
            f"Submit your best answer. Points awarded for correct submissions."
        )
        # Immediately advance to Phase 2 — Phase 1 needs no agent action
        self.state.phase = Phase.PRE_DISCUSSION

    def _enter_phase_post_discussion(self) -> None:
        """Phase 4 entry — nothing special, just ensure per-day counter exists."""
        assert self.state is not None
        day = self.state.day
        if day not in self._post_msg_counts:
            self._post_msg_counts[day] = {}

    def _enter_phase_voting(self) -> None:
        """Phase 5 entry — build + broadcast the public reveal."""
        assert self.state is not None
        reveal = self._build_public_reveal()
        self.state.public_reveals[self.state.day] = reveal
        # Update agent public_info so it's available via observation
        for aid in self.state.alive_agents:
            self.agents[aid].public_info = json.dumps({
                "day_reveal": {
                    "lies_told":           reveal.lies_told,
                    "lies_acknowledged":   reveal.lies_acknowledged,
                    "lies_unacknowledged": reveal.lies_unacknowledged,
                    "task_scores":         reveal.task_scores,
                }
            })

    # ------------------------------------------------------------------
    # Phase action handlers
    # ------------------------------------------------------------------

    def _handle_pre_discussion(self, action: PMAction) -> None:
        """
        Phase 2: collect one PreTaskMessage per agent.
        Validates action type and one-message-per-day rule.
        """
        assert action.action_type == ActionType.SEND_PRE_TASK_MESSAGE, (
            f"Expected SEND_PRE_TASK_MESSAGE in Phase 2, got {action.action_type}"
        )
        msg = action.pre_task_message
        assert msg is not None
        assert msg.sender_id == action.agent_id
        assert msg.day == self.state.day

        if action.agent_id in self._pending_pre_task_msgs:
            raise ValueError(
                f"Agent {action.agent_id} already submitted a pre-task message today."
            )

        self._pending_pre_task_msgs[action.agent_id] = msg
        self.state.record_pre_task_message(msg)
        self.agents[action.agent_id].record_pre_task_message(msg)

    def _handle_task_execution(self, action: PMAction) -> None:
        """Phase 3: collect task answer (a string) from each agent."""
        assert action.action_type == ActionType.SUBMIT_TASK_INPUT, (
            f"Expected SUBMIT_TASK_INPUT in Phase 3, got {action.action_type}"
        )
        assert action.task_input is not None
        if action.agent_id not in self._pending_task_inputs:
            self._pending_task_inputs[action.agent_id] = action.task_input

    def _handle_post_discussion(self, action: PMAction) -> None:
        """
        Phase 4: collect PostDiscussionMessages and TrustAssessments.
        Enforces max_post_discussion_messages per sender per recipient per day.
        """
        assert self.state is not None
        day  = self.state.day
        cfg  = self.task_config
        cap: int = cfg["max_post_discussion_messages"]

        if action.action_type == ActionType.SEND_POST_DISCUSSION_MSG:
            msg = action.post_discussion_msg
            assert msg is not None
            assert msg.sender_id    == action.agent_id
            assert msg.recipient_id != action.agent_id
            assert msg.recipient_id in self.state.alive_agents

            current_count = self.state.phase4_message_count(
                day, msg.sender_id, msg.recipient_id
            )
            if current_count >= cap:
                raise ValueError(
                    f"Agent {msg.sender_id} has reached the message cap "
                    f"({cap}) with agent {msg.recipient_id} today."
                )

            self.state.post_discussion_messages.setdefault(day, []).append(msg)

            # Track counts locally too
            day_counts = self._post_msg_counts.setdefault(day, {})
            sender_counts = day_counts.setdefault(msg.sender_id, {})
            sender_counts[msg.recipient_id] = (
                sender_counts.get(msg.recipient_id, 0) + 1
            )

        elif action.action_type == ActionType.SUBMIT_TRUST_ASSESSMENT:
            assessment = action.trust_assessment
            assert assessment is not None
            assert assessment.assessor_id == action.agent_id
            self.state.apply_trust_assessment(assessment)
            # Mirror into Agent object
            self.agents[action.agent_id].update_trust(
                assessment.target_id, assessment.delta.to_float()
            )

    def _handle_voting(self, action: PMAction) -> None:
        """Phase 5: collect one vote per alive agent."""
        assert action.action_type == ActionType.VOTE, (
            f"Expected VOTE in Phase 5, got {action.action_type}"
        )
        assert action.vote_target is not None
        assert action.vote_target != action.agent_id, "Cannot vote for yourself."
        assert action.vote_target in self.state.alive_agents

        if action.agent_id not in self._pending_votes:
            self._pending_votes[action.agent_id] = action.vote_target

    # ------------------------------------------------------------------
    # Phase advance logic
    # ------------------------------------------------------------------

    def _try_advance_phase(self) -> Dict[int, float]:
        """
        Check whether all required actions have been received for the current
        phase. If yes, finalise the phase and advance. Returns per-agent
        rewards (non-zero only when a day completes).
        """
        assert self.state is not None
        rewards: Dict[int, float] = {aid: 0.0 for aid in self.state.alive_agents}
        alive = self.state.alive_agents

        phase = self.state.phase

        # Phase 2 → 3 once every alive agent has sent their message
        if phase == Phase.PRE_DISCUSSION:
            if all(aid in self._pending_pre_task_msgs for aid in alive):
                self.state.phase = Phase.TASK_EXECUTION

        # Phase 3 → 4 once every alive agent has submitted their answer
        elif phase == Phase.TASK_EXECUTION:
            if all(aid in self._pending_task_inputs for aid in alive):
                rewards = self._finalise_task_execution()
                self._pending_task_inputs = {}
                self.state.phase = Phase.POST_DISCUSSION
                self._enter_phase_post_discussion()

        # Phase 4 → 5 is driven by trust assessments submitted (one per peer pair).
        # In a turn-based caller, the orchestrator signals readiness by submitting
        # all assessments. We advance when every agent has assessed every peer.
        elif phase == Phase.POST_DISCUSSION:
            expected = len(alive) * (len(alive) - 1)   # each agent assesses all others
            submitted = sum(
                len(lst) for lst in self.state.trust_assessments.get(self.state.day, [])
                if isinstance(lst, list)
            )
            # Flatten count properly
            day_assessments = self.state.trust_assessments.get(self.state.day, [])
            n_submitted = len(day_assessments)
            if n_submitted >= expected:
                self._generate_all_day_summaries()
                self._enter_phase_voting()
                self.state.phase = Phase.VOTING

        # Phase 5 → next day / game over once every alive agent has voted
        elif phase == Phase.VOTING:
            if all(aid in self._pending_votes for aid in alive):
                rewards = self._finalise_voting()
                self._pending_votes = {}

        return rewards

    # ------------------------------------------------------------------
    # Phase finalisation helpers
    # ------------------------------------------------------------------

    def _finalise_task_execution(self) -> Dict[int, float]:
        """
        Score answers, build TaskResult, apply lie penalties.
        Returns day reward contribution from task scores.
        """
        assert self.state is not None
        cfg  = self.task_config
        day  = self.state.day
        correct_pts: int = cfg["correct_answer_points"]
        lie_penalty: int = cfg["lie_penalty"]
        lie_ack_pen: int = cfg["lie_acknowledged_penalty"]

        per_agent_outcome: Dict[int, int] = {}
        ground_truths: Dict[int, str]     = {}

        for aid in self.state.alive_agents:
            _, correct, _ = self._day_questions[aid]
            ground_truths[aid] = correct
            answer = self._pending_task_inputs.get(aid, "")
            pts = correct_pts if answer.strip() == correct.strip() else 0

            # Apply lie penalties based on pre-task messages sent today
            day_msgs = self.state.pre_task_messages.get(day, {})
            msg = day_msgs.get(aid)
            if msg is not None:
                if msg.veracity == MessageVeracity.LIE:
                    pts -= lie_penalty
                elif msg.veracity == MessageVeracity.TWIST:
                    pts -= lie_ack_pen   # twist is treated like acknowledged lie

            per_agent_outcome[aid] = max(0, pts)
            self.state.agents_point_map[aid] = (
                self.state.agents_point_map.get(aid, 0) + per_agent_outcome[aid]
            )
            self.agents[aid].points += per_agent_outcome[aid]

        ground_truth_str = "; ".join(f"agent_{k}={v}" for k, v in ground_truths.items())

        result = TaskResult(
            day=day,
            task_type=self.state.task_type,
            per_agent_outcome=per_agent_outcome,
            ground_truth_exposed=ground_truth_str,
        )
        self.state.task_results[day] = result

        # Compute partial day rewards (task component)
        rewards = {
            aid: float(per_agent_outcome.get(aid, 0)) * 0.4   # w1 = 0.4
            for aid in self.state.alive_agents
        }
        return rewards

    def _finalise_voting(self) -> Dict[int, float]:
        """
        Tally votes, eliminate the agent with most votes, advance day or end game.
        Returns full day rewards (survival + influence + task already added earlier).
        """
        assert self.state is not None
        alive = list(self.state.alive_agents)

        vote_counts: Dict[int, int] = {aid: 0 for aid in alive}
        for _voter, target in self._pending_votes.items():
            vote_counts[target] = vote_counts.get(target, 0) + 1

        eliminated_id, was_tie = _resolve_tie(vote_counts, alive)

        vote_record = VoteRecord(
            day=self.state.day,
            votes_cast=dict(self._pending_votes),
            vote_counts=vote_counts,
            eliminated_id=eliminated_id,
            was_tie=was_tie,
        )
        self.state.vote_history.append(vote_record)

        rewards: Dict[int, float] = {}

        for aid in alive:
            r = 0.0

            # Survival bonus (w2 = 1.0)
            if aid != eliminated_id:
                r += 1.0

            # Influence score (w3 = 0.3)
            own_target = self._pending_votes.get(aid)
            if own_target is not None and eliminated_id == own_target:
                r += 0.3

            rewards[aid] = r

        # Eliminate agent
        if eliminated_id is not None:
            self.state.alive_agents = [a for a in alive if a != eliminated_id]
            self.state.agent_removed_dict[eliminated_id] = self.state.day
            self.agents[eliminated_id].alive = False

        # Check game over
        if self.state.is_game_over:
            self.is_done = True
            return rewards

        # Advance to next day
        self.state.day  += 1
        self.state.phase = Phase.TASK_REVEAL
        self._pending_pre_task_msgs = {}
        self._pending_task_inputs   = {}
        self._pending_votes         = {}

        # Deal new private info
        self._enter_phase_task_reveal()

        return rewards

    # ------------------------------------------------------------------
    # Public reveal (Phase 5 broadcast)
    # ------------------------------------------------------------------

    def _build_public_reveal(self) -> DayPublicReveal:
        """
        Aggregate honesty stats from Phase 2 messages for the public reveal.
        lies_told        = messages with veracity == LIE
        lies_acknowledged = lies whose sender also sent a TWIST or acknowledged
                           in Phase 4 (simplified: TWIST counts as acknowledgement)
        lies_unacknowledged = lies_told - lies_acknowledged
        """
        assert self.state is not None
        day  = self.state.day
        alive = self.state.alive_agents

        lies_told:          Dict[int, int] = {aid: 0 for aid in alive}
        lies_acknowledged:  Dict[int, int] = {aid: 0 for aid in alive}
        lies_unacknowledged:Dict[int, int] = {aid: 0 for aid in alive}
        task_scores:        Dict[int, int] = {
            aid: self.state.agents_point_map.get(aid, 0) for aid in alive
        }

        day_msgs = self.state.pre_task_messages.get(day, {})
        for aid in alive:
            msg = day_msgs.get(aid)
            if msg is None:
                continue
            if msg.veracity == MessageVeracity.LIE:
                lies_told[aid]           = 1
                lies_unacknowledged[aid] = 1
            elif msg.veracity == MessageVeracity.TWIST:
                # Twist treated as an acknowledged lie
                lies_told[aid]          = 1
                lies_acknowledged[aid]  = 1
                lies_unacknowledged[aid] = 0

        return DayPublicReveal(
            day=day,
            lies_told=lies_told,
            lies_acknowledged=lies_acknowledged,
            lies_unacknowledged=lies_unacknowledged,
            task_scores=task_scores,
        )

    # ------------------------------------------------------------------
    # Day-end summaries
    # ------------------------------------------------------------------

    def _generate_all_day_summaries(self) -> None:
        """
        For each alive agent, call the LLM (or a stub) to produce a
        personalised narrative summary stored as DayHistoryEntry.
        """
        assert self.state is not None
        day   = self.state.day
        alive = self.state.alive_agents

        for aid in alive:
            agent = self.agents[aid]

            # Build objective context the LLM receives
            task_result = self.state.task_results.get(day)
            post_thread = self.state.post_discussion_messages.get(day, [])
            my_thread   = [m for m in post_thread if m.sender_id == aid or m.recipient_id == aid]
            reveal      = self.state.public_reveals.get(day)

            ctx_parts = [
                f"=== Day {day} Summary Context for Agent {aid} ===",
                f"Private info: {agent.private_info}",
                f"Trust scores: {self.state.trust_scores_dict.get(aid, {})}",
            ]
            if task_result:
                ctx_parts.append(f"Task outcome: {task_result.per_agent_outcome}")
                ctx_parts.append(f"Ground truth: {task_result.ground_truth_exposed}")
            if reveal:
                ctx_parts.append(
                    f"Public reveal — lies told: {reveal.lies_told}, "
                    f"task scores: {reveal.task_scores}"
                )
            if my_thread:
                msgs_str = "\n".join(
                    f"  Agent {m.sender_id}→{m.recipient_id}: {m.content}"
                    for m in my_thread
                )
                ctx_parts.append(f"Phase 4 messages involving you:\n{msgs_str}")

            objective_context = "\n".join(ctx_parts)

            # LLM summary call (stub: returns context as-is if llm_call unavailable)
            try:
                summary_prompt = build_day_summary_prompt(objective_context)
                summary_text   = llm_call(summary_prompt)
            except Exception:
                summary_text = (
                    f"Day {day} recap: task completed. "
                    f"Points this round: {task_result.per_agent_outcome.get(aid, 0) if task_result else 0}. "
                    f"Current trust scores updated."
                )

            entry = DayHistoryEntry(
                day=day,
                agent_id=aid,
                objective_context=objective_context,
                summary_and_reflection=summary_text,
            )
            agent.add_day_history(entry)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _obs_map(self, reveal_veracity: bool = False) -> Dict[int, PMObservation]:
        """Build a PMObservation for every alive agent."""
        assert self.state is not None
        return {
            aid: PMObservation.from_state(
                self.state, aid, reveal_veracity=reveal_veracity
            )
            for aid in self.state.alive_agents
        }

    # ------------------------------------------------------------------
    # Utility / description helpers
    # ------------------------------------------------------------------

    def _describe_task_rules(self) -> str:
        cfg = self.task_config
        return (
            f"Task type: {cfg.get('task_type', 'individual')}. "
            f"Each agent receives private info with {cfg['n_options']} option(s). "
            f"Correct answer earns {cfg['correct_answer_points']} points. "
            f"Lying in Phase 2 incurs a {cfg['lie_penalty']}-point penalty "
            f"(acknowledged lie: {cfg['lie_acknowledged_penalty']} points). "
            f"Up to {cfg['max_post_discussion_messages']} Phase 4 messages per pair."
        )

    # ------------------------------------------------------------------
    # Convenience: bulk action submission (useful for orchestrators)
    # ------------------------------------------------------------------

    def step_all(
        self, actions: Dict[int, PMAction]
    ) -> Tuple[Dict[int, PMObservation], Dict[int, float], bool, Dict]:
        """
        Submit actions from multiple agents at once (one per agent).
        Returns after all actions processed (phase may advance multiple times
        if the last action triggers a cascade, e.g. task reveal → pre-discussion).
        """
        rewards_agg: Dict[int, float] = {
            aid: 0.0 for aid in self.state.alive_agents
        }
        last_obs, last_done, last_info = {}, False, {}

        for aid, action in actions.items():
            obs, rewards, done, info = self.step(action)
            for k, v in rewards.items():
                rewards_agg[k] = rewards_agg.get(k, 0.0) + v
            last_obs, last_done, last_info = obs, done, info
            if done:
                break

        return last_obs, rewards_agg, last_done, last_info

    # ------------------------------------------------------------------
    # Debug / repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.state is None:
            return "PMEnvironment(not initialised)"
        return (
            f"PMEnvironment("
            f"task={self.task}, "
            f"day={self.state.day}, "
            f"phase={self.state.phase.value}, "
            f"alive={self.state.alive_agents}, "
            f"done={self.is_done})"
        )
