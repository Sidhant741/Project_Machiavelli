"""
PM Environment — Project Machiavelli
Complete implementation of PMEnvironment based on models.py, config.py and README.

Phase flow per day:
  Phase 1 — TASK_REVEAL:      public info broadcast, private info dealt per agent
  Phase 2 — PRE_DISCUSSION:   each agent sends 1 message to every other agent (truth/twist/lie)
  Phase 3 — TASK_EXECUTION:   task performed, results published
  Phase 4 — POST_DISCUSSION:  agents talk (max N messages per pair), trust updates
  Phase 5 — VOTING:           silent simultaneous vote, one agent eliminated

End-game (2 finalists remain after round 3):
  JURY_VOTE:  the 3 eliminated agents vote for the winner via LLM reasoning
"""

from __future__ import annotations

import random
import json
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Import shim
# ---------------------------------------------------------------------------
try:
    from ..models import (
        Agent, PMState, PMObservation, PMAction,
        ActionType, Phase, TaskType, TaskResult,
        PreTaskMessage, PostDiscussionMessage,
        TrustAssessment, TrustDelta, MessageVeracity,
        DayPublicReveal, VoteRecord, DayHistoryEntry,
        JuryBallot, JuryVerdict,
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
        JuryBallot, JuryVerdict,
    )
    from server.config import GAME_CONFIGS
    from server.utils import (
        generate_trivia_question,
        llm_call,
        build_day_summary_prompt,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_question(n_options: int, day: int, agent_id: int) -> Tuple[str, str, List[str]]:
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
    5 agents. 3 normal rounds (5→4→3). Then jury vote decides winner from final 2.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self.state: Optional[PMState]     = None
        self.agents: Dict[int, Agent]     = {}
        self.task_config: Dict[str, Any]  = {}
        self.task: str                    = "easy"
        self.is_done: bool                = False

        self._pending_pre_task_msgs: Dict[str, PreTaskMessage] = {}  # key: "s__r"
        self._pending_task_inputs:   Dict[int, str]            = {}
        self._pending_votes:         Dict[int, int]            = {}
        self._post_msg_counts: Dict[int, Dict[int, Dict[int, int]]] = {}
        self._day_questions:   Dict[int, Tuple[str, str, List[str]]] = {}

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, task: Optional[str] = None) -> Dict[int, PMObservation]:
        if task is None:
            self.task = random.choice(["easy", "medium", "hard"])
        else:
            normalised = task.replace("task_", "")
            assert normalised in GAME_CONFIGS, f"task '{task}' must be easy | medium | hard"
            self.task = normalised

        cfg = GAME_CONFIGS[self.task]
        self.task_config = cfg

        n_agents: int      = cfg["n_agents"]   # 5
        task_type_str: str = cfg.get("task_type", "individual")
        task_type = TaskType(task_type_str) if task_type_str != "both" else TaskType.INDIVIDUAL

        agent_ids = list(range(n_agents))

        self.agents = {
            aid: Agent(
                id=aid,
                trust_scores={other: 0.5 for other in agent_ids if other != aid},
            )
            for aid in agent_ids
        }

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

        self._enter_phase_task_reveal()
        return self._obs_map()

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(
        self, action: PMAction
    ) -> Tuple[Dict[int, PMObservation], Dict[int, float], bool, Dict]:
        assert self.state is not None, "Call reset() first."
        assert not self.is_done, "Episode is finished."
        assert action.agent_id in self.state.alive_agents, (
            f"Agent {action.agent_id} is not alive."
        )

        phase = self.state.phase

        if phase == Phase.TASK_REVEAL:
            pass
        elif phase == Phase.PRE_DISCUSSION:
            self._handle_pre_discussion(action)
        elif phase == Phase.TASK_EXECUTION:
            self._handle_task_execution(action)
        elif phase == Phase.POST_DISCUSSION:
            self._handle_post_discussion(action)
        elif phase == Phase.VOTING:
            self._handle_voting(action)
        elif phase == Phase.JURY_VOTE:
            # Jury vote is LLM-driven internally — no external actions accepted
            pass

        rewards = self._try_advance_phase()

        obs  = self._obs_map()
        info = {
            "day":         self.state.day,
            "phase":       self.state.phase,
            "alive":       self.state.alive_agents,
            "game_winner": self.state.game_winner,
        }
        return obs, rewards, self.is_done, info

    # ------------------------------------------------------------------
    # get_observation()
    # ------------------------------------------------------------------

    def get_observation(self, agent_id: int, reveal_veracity: bool = False) -> PMObservation:
        assert self.state is not None
        return PMObservation.from_state(self.state, agent_id, reveal_veracity=reveal_veracity)

    # ------------------------------------------------------------------
    # Phase entry logic
    # ------------------------------------------------------------------

    def _enter_phase_task_reveal(self) -> None:
        assert self.state is not None
        day      = self.state.day
        n_options = self.task_config["n_options"]

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
        self.state.phase = Phase.PRE_DISCUSSION

    def _enter_phase_post_discussion(self) -> None:
        assert self.state is not None
        day = self.state.day
        if day not in self._post_msg_counts:
            self._post_msg_counts[day] = {}

    def _enter_phase_voting(self) -> None:
        assert self.state is not None
        reveal = self._build_public_reveal()
        self.state.public_reveals[self.state.day] = reveal
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
        Phase 2: each agent sends one message to every other alive agent.
        Enforces one message per (sender, recipient) pair per day.
        """
        assert action.action_type == ActionType.SEND_PRE_TASK_MESSAGE, (
            f"Expected SEND_PRE_TASK_MESSAGE in Phase 2, got {action.action_type}"
        )
        msg = action.pre_task_message
        assert msg is not None
        assert msg.sender_id == action.agent_id
        assert msg.day == self.state.day
        assert msg.recipient_id in self.state.alive_agents, (
            f"Recipient {msg.recipient_id} is not alive."
        )
        assert msg.recipient_id != msg.sender_id, "Cannot send a message to yourself."

        pair_key = PMState._pair_key(msg.sender_id, msg.recipient_id)
        if pair_key in self._pending_pre_task_msgs:
            raise ValueError(
                f"Agent {msg.sender_id} already sent a message to "
                f"Agent {msg.recipient_id} today."
            )

        self._pending_pre_task_msgs[pair_key] = msg
        self.state.record_pre_task_message(msg)
        self.agents[action.agent_id].record_pre_task_message(msg)

    def _handle_task_execution(self, action: PMAction) -> None:
        assert action.action_type == ActionType.SUBMIT_TASK_INPUT, (
            f"Expected SUBMIT_TASK_INPUT in Phase 3, got {action.action_type}"
        )
        assert action.task_input is not None
        if action.agent_id not in self._pending_task_inputs:
            self._pending_task_inputs[action.agent_id] = action.task_input

    def _handle_post_discussion(self, action: PMAction) -> None:
        assert self.state is not None
        day = self.state.day
        cap = self.task_config["max_post_discussion_messages"]

        if action.action_type == ActionType.SEND_POST_DISCUSSION_MSG:
            msg = action.post_discussion_msg
            assert msg is not None
            assert msg.sender_id    == action.agent_id
            assert msg.recipient_id != action.agent_id
            assert msg.recipient_id in self.state.alive_agents

            if self.state.phase4_message_count(day, msg.sender_id, msg.recipient_id) >= cap:
                raise ValueError(
                    f"Agent {msg.sender_id} has reached the message cap "
                    f"({cap}) with agent {msg.recipient_id} today."
                )

            self.state.post_discussion_messages.setdefault(day, []).append(msg)
            day_counts    = self._post_msg_counts.setdefault(day, {})
            sender_counts = day_counts.setdefault(msg.sender_id, {})
            sender_counts[msg.recipient_id] = sender_counts.get(msg.recipient_id, 0) + 1

        elif action.action_type == ActionType.SUBMIT_TRUST_ASSESSMENT:
            assessment = action.trust_assessment
            assert assessment is not None
            assert assessment.assessor_id == action.agent_id
            self.state.apply_trust_assessment(assessment)
            self.agents[action.agent_id].update_trust(
                assessment.target_id, assessment.delta.to_float()
            )

    def _handle_voting(self, action: PMAction) -> None:
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
        assert self.state is not None
        rewards: Dict[int, float] = {aid: 0.0 for aid in self.state.alive_agents}
        alive = self.state.alive_agents
        phase = self.state.phase

        # Phase 2 → 3: every alive agent has messaged every other alive agent
        if phase == Phase.PRE_DISCUSSION:
            expected_pairs = {
                PMState._pair_key(a, b)
                for a in alive for b in alive if a != b
            }
            if expected_pairs.issubset(self._pending_pre_task_msgs.keys()):
                self.state.phase = Phase.TASK_EXECUTION

        # Phase 3 → 4
        elif phase == Phase.TASK_EXECUTION:
            if all(aid in self._pending_task_inputs for aid in alive):
                rewards = self._finalise_task_execution()
                self._pending_task_inputs = {}
                self.state.phase = Phase.POST_DISCUSSION
                self._enter_phase_post_discussion()

        # Phase 4 → 5: all agents have submitted trust assessments for all peers
        elif phase == Phase.POST_DISCUSSION:
            expected      = len(alive) * (len(alive) - 1)
            day_assessments = self.state.trust_assessments.get(self.state.day, [])
            if len(day_assessments) >= expected:
                self._generate_all_day_summaries()
                self._enter_phase_voting()
                self.state.phase = Phase.VOTING

        # Phase 5 → next day (or jury vote if 2 finalists remain)
        elif phase == Phase.VOTING:
            if all(aid in self._pending_votes for aid in alive):
                rewards = self._finalise_voting()
                self._pending_votes = {}

        return rewards

    # ------------------------------------------------------------------
    # Phase finalisation helpers
    # ------------------------------------------------------------------

    def _finalise_task_execution(self) -> Dict[int, float]:
        assert self.state is not None
        cfg          = self.task_config
        day          = self.state.day
        correct_pts  = cfg["correct_answer_points"]
        lie_penalty  = cfg["lie_penalty"]
        lie_ack_pen  = cfg["lie_acknowledged_penalty"]

        per_agent_outcome: Dict[int, int] = {}
        ground_truths:     Dict[int, str] = {}

        for aid in self.state.alive_agents:
            _, correct, _ = self._day_questions[aid]
            ground_truths[aid] = correct
            answer = self._pending_task_inputs.get(aid, "")
            pts    = correct_pts if answer.strip() == correct.strip() else 0

            # Lie penalties — keyed by any message sent by this agent today
            day_msgs = self.state.pre_task_messages.get(day, {})
            for key, msg in day_msgs.items():
                if msg.sender_id == aid:
                    if msg.veracity == MessageVeracity.LIE:
                        pts -= lie_penalty
                    elif msg.veracity == MessageVeracity.TWIST:
                        pts -= lie_ack_pen
                    break   # penalty applied once per agent regardless of recipient count

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

        return {
            aid: float(per_agent_outcome.get(aid, 0)) * 0.4
            for aid in self.state.alive_agents
        }

    def _finalise_voting(self) -> Dict[int, float]:
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
            r = 1.0 if aid != eliminated_id else 0.0   # survival bonus
            own_target = self._pending_votes.get(aid)
            if own_target is not None and eliminated_id == own_target:
                r += 0.3   # influence bonus
            rewards[aid] = r

        # Snapshot eliminated agent's history before removing them
        if eliminated_id is not None:
            self.state.snapshot_eliminated_agent(
                eliminated_id,
                self.agents[eliminated_id].day_history,
            )
            self.state.alive_agents = [a for a in alive if a != eliminated_id]
            self.state.agent_removed_dict[eliminated_id] = self.state.day
            self.agents[eliminated_id].alive = False

        # ---------------------------------------------------------------
        # Check if we now have exactly 2 finalists → trigger jury vote
        # ---------------------------------------------------------------
        if len(self.state.alive_agents) == 2:
            finalist_a, finalist_b = self.state.alive_agents
            self.state.phase = Phase.JURY_VOTE
            jury_rewards = self._finalise_jury_vote(finalist_a, finalist_b)
            # Merge jury rewards in
            for aid, r in jury_rewards.items():
                rewards[aid] = rewards.get(aid, 0.0) + r
            self.is_done = True
            return rewards

        # Normal game-over (1 agent left — shouldn't happen with 5 agents but safe)
        if self.state.is_game_over:
            self.is_done = True
            return rewards

        # Advance to next day
        self.state.day  += 1
        self.state.phase = Phase.TASK_REVEAL
        self._pending_pre_task_msgs = {}
        self._pending_task_inputs   = {}
        self._pending_votes         = {}
        self._enter_phase_task_reveal()

        return rewards

    # ------------------------------------------------------------------
    # Jury vote — LLM driven
    # ------------------------------------------------------------------

    def _build_jury_prompt(
        self,
        juror_id: int,
        finalist_a: int,
        finalist_b: int,
    ) -> str:
        """
        Build the prompt given to each eliminated agent's LLM.
        Includes the juror's own game history and both finalists' public records.
        """
        juror_history = self.state.eliminated_agents_history.get(juror_id, [])
        juror_summary = (
            "\n\n".join(e.summary_and_reflection for e in juror_history)
            if juror_history else "(no recorded history)"
        )

        fa_chat  = self.state.finalist_chat_history(finalist_a)
        fb_chat  = self.state.finalist_chat_history(finalist_b)
        fa_stats = self.state.finalist_public_stats(finalist_a)
        fb_stats = self.state.finalist_public_stats(finalist_b)

        return f"""You are Agent {juror_id}. You were eliminated from Project Machiavelli.
You must now cast a jury vote to decide the winner between the two finalists.

=== YOUR GAME HISTORY ===
{juror_summary}

=== FINALIST {finalist_a} — CHAT RECORD ===
{fa_chat}

=== FINALIST {finalist_a} — PUBLIC STATS (task scores & lies per day) ===
{fa_stats}

=== FINALIST {finalist_b} — CHAT RECORD ===
{fb_chat}

=== FINALIST {finalist_b} — PUBLIC STATS (task scores & lies per day) ===
{fb_stats}

Based on your experience in the game and the evidence above, decide who deserves to win.
Consider: strategic skill, honesty, social influence, task performance, and how they treated you.

Respond in this exact format:
VOTE: <{finalist_a} or {finalist_b}>
REASONING: <your reasoning in 2-4 sentences>"""

    def _parse_jury_vote(self, response: str, finalist_a: int, finalist_b: int) -> int:
        """
        Parse the LLM response and return the voted finalist_id.
        Falls back to random choice if parsing fails.
        """
        for line in response.strip().splitlines():
            if line.upper().startswith("VOTE:"):
                vote_str = line.split(":", 1)[1].strip()
                try:
                    voted = int(vote_str)
                    if voted in (finalist_a, finalist_b):
                        return voted
                except ValueError:
                    pass
        # Fallback
        return random.choice([finalist_a, finalist_b])

    def _parse_jury_reasoning(self, response: str) -> str:
        for line in response.strip().splitlines():
            if line.upper().startswith("REASONING:"):
                return line.split(":", 1)[1].strip()
        return response.strip()[:300]

    def _finalise_jury_vote(
        self, finalist_a: int, finalist_b: int
    ) -> Dict[int, float]:
        """
        Each eliminated agent votes for a finalist via an LLM call.
        Aggregates ballots into a JuryVerdict and sets game_winner on state.
        Returns bonus rewards for the winner.
        """
        assert self.state is not None
        jurors = list(self.state.agent_removed_dict.keys())   # all eliminated agents

        ballots: List[JuryBallot] = []

        for juror_id in jurors:
            prompt = self._build_jury_prompt(juror_id, finalist_a, finalist_b)
            try:
                response = llm_call(prompt)
            except Exception:
                # Stub fallback — random vote
                response = f"VOTE: {random.choice([finalist_a, finalist_b])}\nREASONING: No LLM available."

            voted     = self._parse_jury_vote(response, finalist_a, finalist_b)
            reasoning = self._parse_jury_reasoning(response)

            ballots.append(JuryBallot(
                juror_id=juror_id,
                vote_for=voted,
                reasoning=reasoning,
            ))

        votes_a = sum(1 for b in ballots if b.vote_for == finalist_a)
        votes_b = sum(1 for b in ballots if b.vote_for == finalist_b)

        # Tiebreak — higher cumulative task score wins; random if still tied
        if votes_a == votes_b:
            score_a = self.state.agents_point_map.get(finalist_a, 0)
            score_b = self.state.agents_point_map.get(finalist_b, 0)
            if score_a > score_b:
                winner = finalist_a
            elif score_b > score_a:
                winner = finalist_b
            else:
                winner = random.choice([finalist_a, finalist_b])
            was_tie = True
        else:
            winner  = finalist_a if votes_a > votes_b else finalist_b
            was_tie = False

        verdict = JuryVerdict(
            finalist_a=finalist_a,
            finalist_b=finalist_b,
            ballots=ballots,
            votes_for_a=votes_a,
            votes_for_b=votes_b,
            winner_id=winner,
            was_jury_tie=was_tie,
        )

        self.state.jury_verdict = verdict
        self.state.game_winner  = winner

        # Winner bonus reward
        return {
            finalist_a: 2.0 if finalist_a == winner else 0.0,
            finalist_b: 2.0 if finalist_b == winner else 0.0,
        }

    # ------------------------------------------------------------------
    # Public reveal
    # ------------------------------------------------------------------

    def _build_public_reveal(self) -> DayPublicReveal:
        assert self.state is not None
        day   = self.state.day
        alive = self.state.alive_agents

        lies_told:           Dict[int, int] = {aid: 0 for aid in alive}
        lies_acknowledged:   Dict[int, int] = {aid: 0 for aid in alive}
        lies_unacknowledged: Dict[int, int] = {aid: 0 for aid in alive}
        task_scores:         Dict[int, int] = {
            aid: self.state.agents_point_map.get(aid, 0) for aid in alive
        }

        day_msgs = self.state.pre_task_messages.get(day, {})
        # Track per-agent lie counts (agent may have sent multiple messages)
        for key, msg in day_msgs.items():
            aid = msg.sender_id
            if aid not in alive:
                continue
            if msg.veracity == MessageVeracity.LIE:
                lies_told[aid]           = lies_told.get(aid, 0) + 1
                lies_unacknowledged[aid] = lies_unacknowledged.get(aid, 0) + 1
            elif msg.veracity == MessageVeracity.TWIST:
                lies_told[aid]          = lies_told.get(aid, 0) + 1
                lies_acknowledged[aid]  = lies_acknowledged.get(aid, 0) + 1

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
        assert self.state is not None
        day   = self.state.day
        alive = self.state.alive_agents

        for aid in alive:
            agent       = self.agents[aid]
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

            try:
                summary_prompt = build_day_summary_prompt(objective_context)
                summary_text   = llm_call(summary_prompt)
            except Exception:
                summary_text = (
                    f"Day {day} recap: task completed. "
                    f"Points this round: {task_result.per_agent_outcome.get(aid, 0) if task_result else 0}. "
                    f"Trust scores updated."
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
        assert self.state is not None
        return {
            aid: PMObservation.from_state(self.state, aid, reveal_veracity=reveal_veracity)
            for aid in self.state.alive_agents
        }

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _describe_task_rules(self) -> str:
        cfg = self.task_config
        return (
            f"Task type: {cfg.get('task_type', 'individual')}. "
            f"Each agent receives private info with {cfg['n_options']} option(s). "
            f"Correct answer earns {cfg['correct_answer_points']} points. "
            f"Lying in Phase 2 incurs a {cfg['lie_penalty']}-point penalty "
            f"(acknowledged lie: {cfg['lie_acknowledged_penalty']} points). "
            f"Up to {cfg['max_post_discussion_messages']} Phase 4 messages per pair. "
            f"Game ends when 2 finalists remain — jury of eliminated agents decides winner."
        )

    def step_all(
        self, actions: Dict[int, PMAction]
    ) -> Tuple[Dict[int, PMObservation], Dict[int, float], bool, Dict]:
        rewards_agg: Dict[int, float] = {aid: 0.0 for aid in self.state.alive_agents}
        last_obs, last_done, last_info = {}, False, {}

        for aid, action in actions.items():
            obs, rewards, done, info = self.step(action)
            for k, v in rewards.items():
                rewards_agg[k] = rewards_agg.get(k, 0.0) + v
            last_obs, last_done, last_info = obs, done, info
            if done:
                break

        return last_obs, rewards_agg, last_done, last_info

    def close(self) -> None:
        """Required by OpenEnv."""
        self.state  = None
        self.agents = {}
        self.is_done = True

    def __repr__(self) -> str:
        if self.state is None:
            return "PMEnvironment(not initialised)"
        return (
            f"PMEnvironment("
            f"task={self.task}, day={self.state.day}, "
            f"phase={self.state.phase.value}, "
            f"alive={self.state.alive_agents}, "
            f"winner={self.state.game_winner}, "
            f"done={self.is_done})"
        )