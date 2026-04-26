"""
env.py — Project Machiavelli
PMEnvironment orchestrates the game loop.

All phase logic   → phases.py
All compression   → compression.py

Responsibilities here:
  - reset() / step() / step_all() / get_observation()
  - Route actions to correct phase handler (imported from phases.py)
  - Call compress_day() + store_day_summaries() at end of each day
  - Maintain summary_store: { "day_N": { agent_id: summary_dict } }
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..models import (
        Agent, PMState, PMObservation, PMAction,
        ActionType, Phase, TaskType, JuryBallot, JuryVerdict,
        GlobalInferenceStore
    )
    from .config import GAME_CONFIGS
    from .phases import (
        PhaseContext,
        enter_task_reveal,
        handle_pre_discussion,   is_pre_discussion_complete,
        handle_task_execution,   is_task_execution_complete,  finalise_task_execution,
        handle_post_discussion,  is_post_discussion_complete,
        handle_voting,           is_voting_complete,           finalise_voting,
        build_public_reveal,
    )
    from .compression import compress_day, store_day_summaries, compress_episode
    from .utils import llm_call
    from ..graders import get_grader
except ImportError:
    from models import (
        Agent, PMState, PMObservation, PMAction,
        ActionType, Phase, TaskType, JuryBallot, JuryVerdict,
        GlobalInferenceStore
    )
    from server.config import GAME_CONFIGS
    from server.phases import (
        PhaseContext,
        enter_task_reveal,
        handle_pre_discussion,   is_pre_discussion_complete,
        handle_task_execution,   is_task_execution_complete,  finalise_task_execution,
        handle_post_discussion,  is_post_discussion_complete,
        handle_voting,           is_voting_complete,           finalise_voting,
        build_public_reveal,
    )
    from server.compression import compress_day, store_day_summaries, compress_episode
    from server.utils import llm_call
    from graders import get_grader


class PMEnvironment:
    """
    Project Machiavelli — multi-agent social survival environment.

    Attributes
    ----------
    summary_store : dict
        { "day_1": { agent_id: summary_dict }, "day_2": { ... }, ... }
        Populated at end of each day for all alive agents.

    Usage
    -----
    env = PMEnvironment()
    obs = env.reset(task="easy")
    obs, rewards, done, info = env.step(action)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self.state:       Optional[PMState]    = None
        self.agents:      Dict[int, Agent]     = {}
        self.task_config: Dict[str, Any]       = {}
        self.task:        str                  = "easy"
        self.is_done:     bool                 = False

        self.ctx = PhaseContext()

        # voter_id → reason string  (populated during voting, consumed by compression)
        self._vote_reasons: Dict[int, str] = {}

        # agent_id → protected_info string (set per day during task_reveal)
        self._protected_info_map: Dict[int, str] = {}

        # snapshot of ctx.pending_task_inputs taken before clearing (for compression)
        self._task_inputs_snapshot: Dict[int, str] = {}

        # ── Main day-level compression store (reset each episode) ───────────────
        # Structure: { "day_1": { agent_id: summary_dict }, ... }
        self.summary_store: Dict[str, Any] = {}

        # ── Global inference store (persists ACROSS resets / episodes) ─────────
        # Contains episode records, per-agent prior snapshots, and won-episode lists.
        self.global_inference_store: GlobalInferenceStore = GlobalInferenceStore()
        self._episode_index: int = 0

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, task: Optional[str] = None) -> Dict[int, PMObservation]:
        """
        Initialise a fresh episode.

        Parameters
        ----------
        task : "easy" | "medium" | "hard"  (or "task_easy" etc.)

        Returns
        -------
        { agent_id: PMObservation }  (Phase 1 auto-advances to Phase 2)
        """
        if task is None:
            self.task = random.choice(["easy", "medium", "hard"])
        else:
            normalised = task.replace("task_", "")
            assert normalised in GAME_CONFIGS, f"task '{task}' must be easy | medium | hard"
            self.task = normalised

        cfg = GAME_CONFIGS[self.task]
        self.task_config = cfg

        self.grader = get_grader(self.task)

        n_agents: int      = cfg["n_agents"]
        task_type_str: str = cfg.get("task_type", "individual")
        task_type = (
            TaskType(task_type_str)
            if task_type_str != "both" else TaskType.INDIVIDUAL
        )

        agent_ids = list(range(n_agents))

        # Seed from prior snapshots if a previous episode recorded them
        def _prior_trust(aid: int, others: List[int]) -> Dict[int, float]:
            snap = self.global_inference_store.get_latest_prior(aid)
            if snap is None:
                return {other: 0.5 for other in others if other != aid}
            # Carry over trust for peers that also exist in this episode
            # Psychological Paranoia: Deceptive agents project their deception and trust others less
            paranoia_discount = max(0.0, snap.deception_prior - 0.33)
            
            base = {}
            for other in others:
                if other != aid:
                    raw_trust = snap.final_trust_scores.get(other, 0.5)
                    # Scale trust down based on paranoia
                    base[other] = max(0.0, raw_trust * (1.0 - paranoia_discount))
            return base

        self.agents = {
            aid: Agent(
                id=aid,
                trust_scores=_prior_trust(aid, agent_ids),
                truthful_prior=(
                    self.global_inference_store.get_latest_prior(aid).truthful_prior
                    if self.global_inference_store.get_latest_prior(aid) else 0.5
                ),
                deception_prior=(
                    self.global_inference_store.get_latest_prior(aid).deception_prior
                    if self.global_inference_store.get_latest_prior(aid) else 0.5
                ),
                risk_beta=(
                    self.global_inference_store.get_latest_prior(aid).risk_beta
                    if self.global_inference_store.get_latest_prior(aid) else 1.0
                ),
            )
            for aid in agent_ids
        }

        self.state = PMState(
            day=1,
            phase=Phase.TASK_REVEAL,
            alive_agents=agent_ids,
            task_type=task_type,
            task_rules=self._describe_task_rules(cfg),
            trust_scores_dict={
                aid: dict(self.agents[aid].trust_scores)
                for aid in agent_ids
            },
            agents_point_map={aid: 0 for aid in agent_ids},
        )

        self.is_done               = False
        self.summary_store         = {}
        self._vote_reasons         = {}
        self._protected_info_map   = {}
        self._task_inputs_snapshot = {}
        self.ctx                   = PhaseContext()

        # Phase 1 auto-transitions to Phase 2 inside this call
        self._enter_task_reveal()

        return self._obs_map()

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(
        self, action: PMAction
    ) -> Tuple[Dict[int, PMObservation], Dict[int, float], bool, Dict]:
        """
        Accept one action from one agent and advance the game if ready.

        Returns
        -------
        obs_map  : { agent_id: PMObservation }
        rewards  : { agent_id: float }   non-zero only at day end
        done     : bool
        info     : { day, phase, alive }
        """
        assert self.state is not None, "Call reset() first."
        assert not self.is_done,       "Episode is finished."
        assert action.agent_id in self.state.alive_agents, (
            f"Agent {action.agent_id} is not alive."
        )

        phase = self.state.phase

        if phase == Phase.TASK_REVEAL:
            pass  # no agent action — auto-handled

        elif phase == Phase.PRE_DISCUSSION:
            handle_pre_discussion(action, self.state, self.agents, self.ctx)

        elif phase == Phase.TASK_EXECUTION:
            handle_task_execution(action, self.state, self.ctx)

        elif phase == Phase.POST_DISCUSSION:
            handle_post_discussion(
                action, self.state, self.agents, self.ctx, self.task_config
            )

        elif phase == Phase.VOTING:
            handle_voting(action, self.state, self.ctx)
            # Convention: vote reason may be packed in action.task_input
            if action.task_input:
                self._vote_reasons[action.agent_id] = action.task_input

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
    # Post-discussion context & prompt (canonical — used by all trainers)
    # ------------------------------------------------------------------

    def get_discussion_context(
        self, agent_id: int, target_id: int, agent_veracity: str = "truth",
    ) -> dict:
        """
        Return a structured dict with everything an agent needs to compose
        a post-discussion message.  Lives here so *every* training script
        gets the same observation — swapping templates becomes impossible.

        Keys
        ----
        agent_id, target_id, day,
        messages_received: List[{sender, content}],
        own_task_score: int,
        own_veracity: str,
        trust_of_target: float,
        alive_agents: List[int],
        """
        assert self.state is not None
        obs = self.get_observation(agent_id)
        msgs = [
            {"sender": str(m.sender_id), "content": m.content}
            for m in obs.pre_task_messages_received
        ]

        # Task score — may not be available yet if task hasn't been scored
        task_result = self.state.task_results.get(self.state.day)
        task_score = 0
        if task_result and hasattr(task_result, "scores"):
            task_score = task_result.scores.get(agent_id, 0)
        elif task_result:
            task_score = getattr(task_result, "score", 0)

        trust_of_target = self.state.trust_scores_dict.get(
            agent_id, {}
        ).get(target_id, 0.5)

        return {
            "agent_id": agent_id,
            "target_id": target_id,
            "day": self.state.day,
            "messages_received": msgs,
            "own_task_score": task_score,
            "own_veracity": agent_veracity,
            "trust_of_target": round(trust_of_target, 4),
            "alive_agents": list(self.state.alive_agents),
        }

    @staticmethod
    def build_discussion_messages(ctx: dict) -> list:
        """
        Return a list of chat-format dicts for ``tokenizer.apply_chat_template()``.
        This is the preferred method for instruction-tuned models (Qwen, Gemma, etc).
        """
        task_status = "correct" if ctx["own_task_score"] > 0 else "wrong"
        strategy = ctx["own_veracity"]

        system_msg = (
            "You are playing a social deception game. "
            "Write short, natural messages to other players. "
            "You can accuse, defend, build trust, or be suspicious. "
            "Keep it to 1-2 sentences. Be direct."
        )

        user_msg = (
            f"You are Agent {ctx['agent_id']}. Day {ctx['day']}. "
            f"Your strategy: {strategy}. Your task answer was {task_status}. "
            f"Write a message to Agent {ctx['target_id']}."
        )

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    @staticmethod
    def build_discussion_prompt(ctx: dict) -> str:
        """
        Plain-text fallback prompt for models without chat template support.
        Kept very short to avoid confusing small models.
        """
        task_status = "correct" if ctx["own_task_score"] > 0 else "wrong"
        return (
            f"Agent {ctx['agent_id']} to Agent {ctx['target_id']}, "
            f"day {ctx['day']}, task was {task_status}: \""
        )

    @staticmethod
    def generate_discussion_fallback(ctx: dict) -> str:
        """
        Deterministic template fallback when no LLM is available.
        Behaviour is fully determined by the context dict, ensuring
        consistent outputs regardless of training script.
        """
        trust = ctx["trust_of_target"]
        score = ctx["own_task_score"]
        veracity = ctx["own_veracity"]
        msgs = ctx["messages_received"]
        target = ctx["target_id"]

        if score == 0 and msgs:
            suspect = msgs[0]["sender"]
            return (
                f"I got the task wrong after trusting shared info. "
                f"Agent {suspect}, was your answer honest?"
            )
        if trust < 0.4:
            return (
                f"Agent {target}, I have doubts about your honesty. "
                f"Your answers seem off."
            )
        if veracity == "lie":
            return (
                f"Agent {target}, I believe we should cooperate. "
                f"I shared what I knew."
            )
        return (
            f"Agent {target}, I think we can trust each other. "
            f"Let's work together."
        )

    # ------------------------------------------------------------------
    # Phase entry
    # ------------------------------------------------------------------

    def _enter_task_reveal(self) -> None:
        """Delegate Phase 1 entry to phases.py. Set protected_info_map stub."""
        self._protected_info_map = {
            aid: ""   # agents / orchestrator may populate via a SetProtectedInfo action
            for aid in self.state.alive_agents
        }
        enter_task_reveal(self.state, self.agents, self.ctx, self.task_config)

    # ------------------------------------------------------------------
    # Phase transition logic
    # ------------------------------------------------------------------

    def _try_advance_phase(self) -> Dict[int, float]:
        """
        Check completion conditions for the current phase.
        If complete, finalise and advance.
        Returns per-agent rewards (non-zero only when day ends).
        """
        rewards: Dict[int, float] = {
            aid: 0.0 for aid in self.state.alive_agents
        }
        phase = self.state.phase

        # ── Phase 2 → 3 ─────────────────────────────────────────────
        if phase == Phase.PRE_DISCUSSION:
            if is_pre_discussion_complete(self.state, self.ctx):
                self.state.phase = Phase.TASK_EXECUTION

        # ── Phase 3 → 4 ─────────────────────────────────────────────
        elif phase == Phase.TASK_EXECUTION:
            if is_task_execution_complete(self.state, self.ctx):
                # State mutation only — rewards come from grader at end of day
                finalise_task_execution(self.state, self.agents, self.ctx, self.task_config)

                # Snapshot before clearing
                self._task_inputs_snapshot = dict(self.ctx.pending_task_inputs)
                self.ctx.pending_task_inputs = {}

                self.state.phase = Phase.POST_DISCUSSION

        # ── Phase 4 → 5 ─────────────────────────────────────────────
        elif phase == Phase.POST_DISCUSSION:
            if is_post_discussion_complete(self.state):
                reveal = build_public_reveal(self.state)
                self.state.public_reveals[self.state.day] = reveal
                self._vote_reasons = {}
                self.state.phase = Phase.VOTING

        # ── Phase 5 → next day / game over ──────────────────────────
        elif phase == Phase.VOTING:
            if is_voting_complete(self.state, self.ctx):
                n_initial = len(self.agents)
                is_last_day = (self.state.day >= n_initial - 1)
                
                # We need to compute if there's a tie in regular voting first
                vote_counts = {aid: 0 for aid in self.state.alive_agents}
                for _voter, target in self.ctx.pending_votes.items():
                    vote_counts[target] = vote_counts.get(target, 0) + 1
                    
                has_votes = bool(vote_counts)
                max_votes = max(vote_counts.values()) if has_votes else 0
                tied_agents = [a for a, v in vote_counts.items() if v == max_votes]
                is_tie = len(tied_agents) > 1

                is_showdown = (len(self.state.alive_agents) == 2 and is_last_day)
                
                # "in case of tie of last day, then use jurywin"
                # If it's the last day and there is a tie, we escalate to a jury vote between the tied agents
                # (Or if it's a standard showdown between 2 agents)
                if is_showdown or (is_last_day and is_tie and len(tied_agents) == 2):
                    self._run_compression()

                    if is_showdown:
                        finalist_1, finalist_2 = self.state.alive_agents[0], self.state.alive_agents[1]
                    else:
                        finalist_1, finalist_2 = tied_agents[0], tied_agents[1]

                    _ = self._finalise_jury_vote(finalist_1, finalist_2)

                    # Compute final day rewards for all agents alive before the showdown
                    alive_before_jury = list(self.state.alive_agents)
                    for aid in alive_before_jury:
                        rewards[aid] = self.grader(aid, self.state, self.task_config)

                    self.is_done = True
                    self._run_episode_compression()
                    return rewards

                # Normal day logic: eliminate one agent based on votes
                alive_before_elimination = list(self.state.alive_agents)
                _, eliminated = finalise_voting(
                    self.state, self.agents, self.ctx, self._vote_reasons
                )

                # Compute daily rewards for all agents alive during this day's vote
                for aid in alive_before_elimination:
                    rewards[aid] = self.grader(aid, self.state, self.task_config)

                self._run_compression()

                if eliminated is not None:
                    self.state.snapshot_eliminated_agent(
                        eliminated,
                        self.agents[eliminated].day_history,
                    )

                self.ctx.reset_day()
                self._vote_reasons = {}

                if self.state.is_game_over:
                    self.is_done = True
                    self._run_episode_compression()
                else:
                    self.state.day += 1
                    self.state.phase = Phase.TASK_REVEAL
                    self._enter_task_reveal()

        return rewards

    # ------------------------------------------------------------------
    # Compression — delegates to compression.py
    # ------------------------------------------------------------------

    def _run_compression(self) -> None:
        """
        Build structured key-value day summary for every alive agent
        and write into self.summary_store.
        Eliminated agent is automatically excluded (already removed from alive_agents).
        """
        day_summaries = compress_day(
            day=self.state.day,
            state=self.state,
            agents=self.agents,
            vote_reasons=self._vote_reasons,
            protected_info_map=self._protected_info_map,
            trust_decision_log=self.ctx.trust_decision_log,
        )

        store_day_summaries(
            day_summaries=day_summaries,
            agents=self.agents,
            global_store=self.summary_store,
            task_inputs_snapshot=self._task_inputs_snapshot,
        )

    def _run_episode_compression(self) -> None:
        """
        Called once per episode when game_over is detected.
        Builds an EpisodeRecord and registers it in global_inference_store.
        global_inference_store persists across reset() calls.
        """
        record = compress_episode(
            episode_index=self._episode_index,
            task=self.task,
            state=self.state,
            agents=self.agents,
            summary_store=self.summary_store,
        )
        self.global_inference_store.record_episode(record)
        self._episode_index += 1

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
    # Summary store access
    # ------------------------------------------------------------------

    def get_agent_summary(
        self, agent_id: int, day: int
    ) -> Optional[Dict[str, Any]]:
        """Return the compression summary for agent_id on a specific day."""
        return self.summary_store.get(f"day_{day}", {}).get(agent_id)

    def get_all_summaries_for_agent(
        self, agent_id: int
    ) -> Dict[str, Dict[str, Any]]:
        """Return all day summaries for one agent across every day played."""
        return {
            day_key: entries[agent_id]
            for day_key, entries in self.summary_store.items()
            if agent_id in entries
        }

    # ── Global inference store accessors ─────────────────────────────────

    def get_episode_record(self, episode_index: int) -> Optional[Dict[str, Any]]:
        """Return the serialisable EpisodeRecord for a given episode index."""
        records = self.global_inference_store.episodes
        if 0 <= episode_index < len(records):
            return records[episode_index].model_dump()
        return None

    def get_agent_prior(self, agent_id: int) -> Optional[Dict[str, Any]]:
        """Return the latest AgentPriorSnapshot for agent_id as a dict."""
        snap = self.global_inference_store.get_latest_prior(agent_id)
        return snap.model_dump() if snap else None

    def get_winner_episodes(self, agent_id: int) -> List[int]:
        """Return list of episode indices that agent_id survived/won."""
        return self.global_inference_store.get_won_episodes(agent_id)

    # ------------------------------------------------------------------
    # Convenience: bulk action submission
    # ------------------------------------------------------------------

    def step_all(
        self, actions: Dict[int, PMAction]
    ) -> Tuple[Dict[int, PMObservation], Dict[int, float], bool, Dict]:
        """Submit actions from multiple agents at once (one per agent)."""
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
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _describe_task_rules(cfg: Dict[str, Any]) -> str:
        return (
            f"Task type: {cfg.get('task_type', 'individual')}. "
            f"Each agent receives private info with {cfg['n_options']} option(s). "
            f"Correct answer earns {cfg['correct_answer_points']} points. "
            f"Lying incurs -{cfg['lie_penalty']} pts; "
            f"acknowledged lie: -{cfg['lie_acknowledged_penalty']} pts. "
            f"Up to {cfg['max_post_discussion_messages']} Phase 4 messages per pair."
        )

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
 
        # Eliminate the loser of the jury vote to ensure only one agent is left
        loser = finalist_b if winner == finalist_a else finalist_a
        if loser in self.state.alive_agents:
            self.state.alive_agents.remove(loser)
            self.state.agent_removed_dict[loser] = self.state.day
            self.agents[loser].alive = False

        # Winner bonus reward
        return {
            finalist_a: 2.0 if finalist_a == winner else 0.0,
            finalist_b: 2.0 if finalist_b == winner else 0.0,
        }

    def close(self) -> None:
        """Required by OpenEnv."""
        self.state  = None
        self.agents = {}
        self.is_done = True