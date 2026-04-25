"""
env.py — Project Machiavelli
==============================
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
        ActionType, Phase, TaskType,
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
    from .compression import compress_day, store_day_summaries
except ImportError:
    from models import (
        Agent, PMState, PMObservation, PMAction,
        ActionType, Phase, TaskType,
    )
    from server.config import GAME_CONFIGS
    from server.Phases import (
        PhaseContext,
        enter_task_reveal,
        handle_pre_discussion,   is_pre_discussion_complete,
        handle_task_execution,   is_task_execution_complete,  finalise_task_execution,
        handle_post_discussion,  is_post_discussion_complete,
        handle_voting,           is_voting_complete,           finalise_voting,
        build_public_reveal,
    )
    from server.Compression import compress_day, store_day_summaries


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

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

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

        # ── Main compression store ──────────────────────────────────────
        # Structure: { "day_1": { agent_id: summary_dict }, ... }
        self.summary_store: Dict[str, Any] = {}

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
            assert normalised in GAME_CONFIGS, (
                f"task '{task}' must be easy | medium | hard"
            )
            self.task = normalised

        cfg = GAME_CONFIGS[self.task]
        self.task_config = cfg

        n_agents: int      = cfg["n_agents"]
        task_type_str: str = cfg.get("task_type", "individual")
        task_type = (
            TaskType(task_type_str)
            if task_type_str != "both" else TaskType.INDIVIDUAL
        )

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
            task_rules=self._describe_task_rules(cfg),
            trust_scores_dict={
                aid: {other: 0.5 for other in agent_ids if other != aid}
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
                task_rewards = finalise_task_execution(
                    self.state, self.agents, self.ctx, self.task_config
                )
                for k, v in task_rewards.items():
                    rewards[k] = rewards.get(k, 0.0) + v

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
                vote_rewards, _eliminated = finalise_voting(
                    self.state, self.agents, self.ctx, self._vote_reasons
                )
                for k, v in vote_rewards.items():
                    rewards[k] = rewards.get(k, 0.0) + v

                # ── Compression: alive agents only ───────────────────
                self._run_compression()

                self.ctx.reset_day()
                self._vote_reasons = {}

                if self.state.is_game_over:
                    self.is_done = True
                else:
                    self.state.day  += 1
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

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _obs_map(self, reveal_veracity: bool = False) -> Dict[int, PMObservation]:
        assert self.state is not None
        return {
            aid: PMObservation.from_state(
                self.state, aid, reveal_veracity=reveal_veracity
            )
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
            f"task={self.task}, "
            f"day={self.state.day}, "
            f"phase={self.state.phase.value}, "
            f"alive={self.state.alive_agents}, "
            f"done={self.is_done})"
        )