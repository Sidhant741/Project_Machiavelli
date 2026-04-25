"""
smoke_test.py — Project Machiavelli
====================================
Full-game smoke test (easy difficulty) using stub agents.
Runs until game over (1 agent left) so episode-level compression fires.

Run from the project root:
    python test_smoke.py
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(__file__))

from server.environment import PMEnvironment
from models import (
    PMAction, ActionType,
    PreTaskMessage, PostDiscussionMessage, TrustAssessment,
    MessageVeracity, TrustDelta,
)

# ─────────────────────────────────────────────────────────────────────────────
# Stub action builders
# ─────────────────────────────────────────────────────────────────────────────

def build_pre_task_message(sender_id: int, recipient_id: int, day: int) -> PMAction:
    return PMAction(
        agent_id=sender_id,
        action_type=ActionType.SEND_PRE_TASK_MESSAGE,
        pre_task_message=PreTaskMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=f"My answer is {random.randint(1, 100)}",
            veracity=MessageVeracity.TRUTH,
            day=day,
        ),
    )


def build_task_input(agent_id: int, answer: str) -> PMAction:
    return PMAction(
        agent_id=agent_id,
        action_type=ActionType.SUBMIT_TASK_INPUT,
        task_input=answer,
    )


def build_post_msg(sender: int, recipient: int, day: int, turn: int) -> PMAction:
    return PMAction(
        agent_id=sender,
        action_type=ActionType.SEND_POST_DISCUSSION_MSG,
        post_discussion_msg=PostDiscussionMessage(
            sender_id=sender,
            recipient_id=recipient,
            content="Let's talk strategy.",
            day=day,
            turn_index=turn,
        ),
    )


def build_trust_assessment(assessor: int, target: int, day: int) -> PMAction:
    return PMAction(
        agent_id=assessor,
        action_type=ActionType.SUBMIT_TRUST_ASSESSMENT,
        trust_assessment=TrustAssessment(
            assessor_id=assessor,
            target_id=target,
            day=day,
            reasoning="Seemed cooperative.",
            delta=TrustDelta.NEUTRAL,
        ),
    )


def build_vote(voter: int, target: int) -> PMAction:
    return PMAction(
        agent_id=voter,
        action_type=ActionType.VOTE,
        vote_target=target,
        task_input="I don't trust them.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# One full-day helper
# ─────────────────────────────────────────────────────────────────────────────

def play_one_day(env: PMEnvironment) -> bool:
    """
    Execute all 5 phases for the current day.
    Returns True when env.is_done.
    """
    alive = list(env.state.alive_agents)
    day   = env.state.day
    print(f"\n{'─'*50}")
    print(f"  DAY {day}  |  alive={alive}")
    print(f"{'─'*50}")

    # Phase 2
    print("  [P2] PRE_DISCUSSION")
    for sender in alive:
        recipient = alive[(alive.index(sender) + 1) % len(alive)]
        env.step(build_pre_task_message(sender, recipient, day))
    if env.is_done:
        return True

    # Phase 3
    print("  [P3] TASK_EXECUTION")
    for aid in list(env.state.alive_agents):
        env.step(build_task_input(aid, "42"))
    if env.is_done:
        return True

    # Phase 4 — messages then trust assessments
    print("  [P4] POST_DISCUSSION")
    alive4 = list(env.state.alive_agents)
    turn_tracker = {}
    for sender in alive4:
        for recipient in alive4:
            if sender == recipient:
                continue
            key = (sender, recipient)
            turn = turn_tracker.get(key, 0)
            env.step(build_post_msg(sender, recipient, day, turn))
            turn_tracker[key] = turn + 1
    if env.is_done:
        return True

    for assessor in alive4:
        for target in alive4:
            if assessor == target:
                continue
            env.step(build_trust_assessment(assessor, target, day))
    if env.is_done:
        return True

    # Phase 5
    print("  [P5] VOTING")
    alive5 = list(env.state.alive_agents)
    for voter in alive5:
        candidates = [a for a in alive5 if a != voter]
        target = random.choice(candidates)
        _, _, done, _ = env.step(build_vote(voter, target))
        print(f"     agent {voter} → {target}  |  done={done}")
        if done:
            return True

    return env.is_done


# ─────────────────────────────────────────────────────────────────────────────
# Main smoke test
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test():
    print("\n" + "="*60)
    print("  PROJECT MACHIAVELLI — SMOKE TEST  (full game)")
    print("="*60)

    env = PMEnvironment()
    
    # Run 3 consecutive episodes to prove data accumulates
    for ep in range(3):
        print(f"\n🚀 STARTING EPISODE {ep}")
        env.reset(task="easy")
        
        # Play until game over
        while not env.is_done:
            play_one_day(env)

    # ── Day-level compression store ──────────────────────────────────────────
    print("\n── Compression Store (day-level) ───────────────────────────")
    for key, day_store in env.summary_store.items():
        print(f"   {key}: {len(day_store)} agent summaries")
        for aid, summary in day_store.items():
            print(f"     agent {aid}: score={summary['task']['score']}, "
                  f"survived={summary['voting']['survived']}")

    # ── Global Inference Store (episode-level) ───────────────────────────────
    print("\n── Global Inference Store (episode-level) ──────────────────")
    gis = env.global_inference_store
    print(f"   Episodes recorded : {len(gis.episodes)}")

    for ep in gis.episodes:
        print(f"\n   Episode {ep.episode_index}:")
        print(f"     task             = {ep.task}")
        print(f"     days_played      = {ep.days_played}")
        print(f"     winner_ids       = {ep.winner_ids}")
        print(f"     eliminated_order = {ep.eliminated_order}")
        for aid in sorted(ep.agent_day_summaries.keys()):
            day_sums = ep.agent_day_summaries[aid]
            snap     = ep.prior_snapshots.get(aid)
            won_eps  = env.get_winner_episodes(aid)
            print(f"\n     agent {aid}:")
            print(f"       days_alive      = {len(day_sums)}")
            print(f"       won_episodes    = {won_eps}")
            if snap:
                print(f"       truthful_prior  = {snap.truthful_prior}")
                print(f"       deception_prior = {snap.deception_prior}")
                print(f"       risk_beta       = {snap.risk_beta}")
                print(f"       final_trust     = {snap.final_trust_scores}")

    # Assertions
    assert len(gis.episodes) == 3, \
        f"Expected 3 episodes, got {len(gis.episodes)}"
    ep0 = gis.episodes[0]
    assert len(ep0.winner_ids) >= 1, "No winner recorded"
    assert len(ep0.prior_snapshots) == ep0.n_agents, \
        "Prior snapshot missing for some agent"
    for aid in range(ep0.n_agents):
        assert aid in ep0.agent_day_summaries, \
            f"Missing day summaries for agent {aid}"

    print("\n✅  SMOKE TEST PASSED\n")


if __name__ == "__main__":
    run_smoke_test()
