"""
FastAPI + Gradio application for Project Machiavelli.

A multi-agent social survival environment where LLM agents compete
through deception, negotiation, and coalition formation.

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Direct:
    python app.py
"""

import os
import sys
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent.parent
src_dir = str(repo_root / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# ---------------------------------------------------------------------------
# Env imports
# ---------------------------------------------------------------------------
try:
    from ..models import (
        PMAction, PMObservation, ActionType, Phase,
        PreTaskMessage, PostDiscussionMessage,
        TrustAssessment, TrustDelta, MessageVeracity,
    )
    from .environment import PMEnvironment
    from .config import GAME_CONFIGS
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    from models import (
        PMAction, PMObservation, ActionType, Phase,
        PreTaskMessage, PostDiscussionMessage,
        TrustAssessment, TrustDelta, MessageVeracity,
    )
    from server.environment import PMEnvironment
    from server.config import GAME_CONFIGS

# ---------------------------------------------------------------------------
# OpenEnv web interface
# ---------------------------------------------------------------------------
from openenv.core.env_server import create_web_interface_app
import openenv.core.env_server.web_interface as web_interface
from openenv.core.env_server.serialization import serialize_observation

def create_pm_environment():
    return PMEnvironment()

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
import gradio as gr

# Phase display labels
PHASE_LABELS = {
    "task_reveal":    "📋 Phase 1 — Task Reveal",
    "pre_discussion": "💬 Phase 2 — Pre-Discussion",
    "task_execution": "⚙️  Phase 3 — Task Execution",
    "post_discussion":"🔍 Phase 4 — Post-Discussion",
    "voting":         "🗳️  Phase 5 — Voting",
}

def _format_obs(obs_dict: dict) -> str:
    """Convert raw observation dict into a readable Markdown summary."""
    if not obs_dict:
        return "No observation yet. Click **Reset** to start."

    obs = obs_dict.get("observation", obs_dict)
    day   = obs.get("day", "?")
    phase = obs.get("phase", "?")
    alive = obs.get("alive_agents", [])
    phase_label = PHASE_LABELS.get(phase, phase)

    lines = [
        f"## Day {day} · {phase_label}",
        f"**Alive agents:** {', '.join(f'Agent {a}' for a in alive)}",
        "",
    ]

    # Per-agent trust + points
    trust = obs.get("trust_scores", {})
    points = obs.get("own_points", None)
    if points is not None:
        lines.append(f"**Your points:** {points}")
    if trust:
        trust_str = "  ".join(f"→ Agent {k}: `{v:.2f}`" for k, v in trust.items())
        lines.append(f"**Trust scores:** {trust_str}")

    # Private info
    private = obs.get("own_private_info", "")
    if private:
        lines += ["", f"**Your private info:**", f"> {private}"]

    # Public info
    public = obs.get("global_public_info", "")
    if public:
        lines += ["", f"**Public info:**", f"> {public}"]

    # Phase 2 messages received
    msgs = obs.get("pre_task_messages_received", [])
    if msgs:
        lines += ["", "**Messages received (Phase 2):**"]
        for m in msgs:
            sender = m.get("sender_id", "?")
            content = m.get("content", "")
            lines.append(f"- Agent {sender}: *\"{content}\"*")

    # Phase 5 public reveal
    reveal = obs.get("public_reveal")
    if reveal:
        lines += ["", "**📢 Public Reveal:**"]
        lies = reveal.get("lies_told", {})
        scores = reveal.get("task_scores", {})
        for aid in alive:
            aid_str = str(aid)
            lines.append(
                f"- Agent {aid}: lies told = {lies.get(aid_str, lies.get(aid, 0))}, "
                f"task score = {scores.get(aid_str, scores.get(aid, 0))}"
            )

    # Last round votes
    votes = obs.get("votes_last_round", {})
    if votes:
        lines += ["", "**Last round votes:**"]
        for voter, target in votes.items():
            lines.append(f"- Agent {voter} → Agent {target}")

    return "\n".join(lines)


def _format_state(env: PMEnvironment) -> str:
    """Compact state dump for the debug panel."""
    if env is None or env.state is None:
        return "{}"
    s = env.state
    return json.dumps({
        "day":         s.day,
        "phase":       s.phase,
        "alive":       s.alive_agents,
        "points":      s.agents_point_map,
        "trust":       s.trust_scores_dict,
        "eliminated":  s.agent_removed_dict,
        "vote_history": [
            {
                "day":          v.day,
                "votes_cast":   v.votes_cast,
                "eliminated_id": v.eliminated_id,
                "was_tie":      v.was_tie,
            }
            for v in s.vote_history
        ],
    }, indent=2)


def build_gradio_app(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    import openenv.core.env_server.gradio_ui as gradio_ui
    readme_content  = gradio_ui._readme_section(metadata)
    display_title   = gradio_ui.get_gradio_display_title(metadata, fallback=title)

    with gr.Blocks(
        title=display_title,
        css="""
        .col-left  { min-width: 280px; }
        .col-right { min-width: 500px; }
        .phase-badge { font-weight: bold; font-size: 1.1em; }
        """,
    ) as demo:

        env_state = gr.State()

        with gr.Row():
            # ----------------------------------------------------------------
            # LEFT PANEL — info + controls
            # ----------------------------------------------------------------
            with gr.Column(scale=1, elem_classes="col-left"):
                if quick_start_md:
                    with gr.Accordion("Quick Start", open=True):
                        gr.Markdown(quick_start_md)
                with gr.Accordion("README", open=False):
                    gr.Markdown(readme_content)

                gr.Markdown("---")
                gr.Markdown("### 🎮 Game Controls")

                difficulty = gr.Radio(
                    ["easy", "medium", "hard"],
                    label="Difficulty",
                    value="easy",
                )
                reset_btn = gr.Button("🔄 Reset Game", variant="secondary")

                gr.Markdown("---")
                gr.Markdown("### ⚡ Submit Action")

                # Phase selector — tells the UI which action form to show
                phase_selector = gr.Radio(
                    ["Phase 2 — Message", "Phase 3 — Task", "Phase 4 — Discussion", "Phase 4 — Trust", "Phase 5 — Vote"],
                    label="Action type",
                    value="Phase 2 — Message",
                )

                # ---- Phase 2 ------------------------------------------------
                with gr.Group(visible=True) as grp_phase2:
                    gr.Markdown("**Send pre-task message**")
                    p2_recipient   = gr.Number(label="Recipient agent ID", value=1, precision=0)
                    p2_content     = gr.Textbox(label="Message content", placeholder="What will you tell them?")
                    p2_veracity    = gr.Dropdown(
                        choices=["truth", "twist", "lie"],
                        label="Veracity (self-label)",
                        value="truth",
                    )
                    p2_private_ref = gr.Textbox(label="Private info referenced (optional)", placeholder="Snapshot of your private info")
                    p2_btn         = gr.Button("Send Message", variant="primary")

                # ---- Phase 3 ------------------------------------------------
                with gr.Group(visible=False) as grp_phase3:
                    gr.Markdown("**Submit task answer**")
                    p3_answer = gr.Textbox(label="Your answer", placeholder="Enter your answer exactly as shown in options")
                    p3_btn    = gr.Button("Submit Answer", variant="primary")

                # ---- Phase 4 — message --------------------------------------
                with gr.Group(visible=False) as grp_phase4_msg:
                    gr.Markdown("**Send post-discussion message**")
                    p4_recipient = gr.Number(label="Recipient agent ID", value=1, precision=0)
                    p4_content   = gr.Textbox(label="Message content", lines=3)
                    p4_msg_btn   = gr.Button("Send Message", variant="primary")

                # ---- Phase 4 — trust ----------------------------------------
                with gr.Group(visible=False) as grp_phase4_trust:
                    gr.Markdown("**Submit trust assessment**")
                    p4t_target = gr.Number(label="Target agent ID", value=1, precision=0)
                    p4t_delta  = gr.Dropdown(
                        choices=["strong_increase", "increase", "neutral", "decrease", "strong_decrease"],
                        label="Trust delta",
                        value="neutral",
                    )
                    p4t_reason = gr.Textbox(label="Reasoning (private)", lines=2, placeholder="Why are you updating trust?")
                    p4t_btn    = gr.Button("Submit Assessment", variant="primary")

                # ---- Phase 5 ------------------------------------------------
                with gr.Group(visible=False) as grp_phase5:
                    gr.Markdown("**Cast your vote**")
                    p5_target = gr.Number(label="Vote target agent ID", value=1, precision=0)
                    p5_btn    = gr.Button("Cast Vote", variant="primary")

                state_btn = gr.Button("🔍 Get Full State", variant="secondary")

            # ----------------------------------------------------------------
            # RIGHT PANEL — observation + output
            # ----------------------------------------------------------------
            with gr.Column(scale=2, elem_classes="col-right"):
                obs_display = gr.Markdown(
                    value="## Welcome to Project Machiavelli\n\nChoose a difficulty and click **Reset Game** to start.\n\n"
                          "*Agents will compete through deception, negotiation, and coalition formation.*"
                )
                status  = gr.Textbox(label="Status", interactive=False)
                raw_json = gr.Code(label="Raw JSON / State", language="json", interactive=False)

        # --------------------------------------------------------------------
        # Phase selector visibility logic
        # --------------------------------------------------------------------
        def update_phase_groups(phase_sel):
            return [
                gr.update(visible=(phase_sel == "Phase 2 — Message")),
                gr.update(visible=(phase_sel == "Phase 3 — Task")),
                gr.update(visible=(phase_sel == "Phase 4 — Discussion")),
                gr.update(visible=(phase_sel == "Phase 4 — Trust")),
                gr.update(visible=(phase_sel == "Phase 5 — Vote")),
            ]

        phase_selector.change(
            fn=update_phase_groups,
            inputs=[phase_selector],
            outputs=[grp_phase2, grp_phase3, grp_phase4_msg, grp_phase4_trust, grp_phase5],
        )

        # --------------------------------------------------------------------
        # Reset
        # --------------------------------------------------------------------
        def do_reset(difficulty, current_env):
            if current_env is None:
                current_env = create_pm_environment()
            try:
                obs_map = current_env.reset(task=difficulty)
                # Show observation for agent 0 by default
                obs0 = obs_map.get(0)
                obs_dict = obs0.model_dump() if obs0 else {}
                return (
                    _format_obs(obs_dict),
                    f"✅ Game reset — difficulty: {difficulty}, {len(obs_map)} agents alive.",
                    "",
                    current_env,
                )
            except Exception as exc:
                import traceback; traceback.print_exc()
                return "", f"❌ Error: {exc}", "", current_env

        reset_btn.click(
            fn=do_reset,
            inputs=[difficulty, env_state],
            outputs=[obs_display, status, raw_json, env_state],
        )

        # --------------------------------------------------------------------
        # Step helpers
        # --------------------------------------------------------------------
        def _step(env: PMEnvironment, action: PMAction):
            obs_map, rewards, done, info = env.step(action)
            obs0 = obs_map.get(0) or next(iter(obs_map.values()), None)
            obs_dict = obs0.model_dump() if obs0 else {}
            reward_str = ", ".join(f"Agent {k}: {v:+.2f}" for k, v in rewards.items())
            status_msg = (
                f"✅ Action submitted | Phase: {info['phase'].value} | "
                f"Day: {info['day']} | Alive: {info['alive']} | "
                f"Rewards: [{reward_str}]"
                + (" | 🏁 GAME OVER" if done else "")
            )
            return _format_obs(obs_dict), status_msg, ""

        # Phase 2
        def do_phase2(env, agent_id, recipient, content, veracity, private_ref):
            if env is None:
                return "", "❌ Reset first.", ""
            try:
                action = PMAction(
                    agent_id=int(agent_id) if agent_id is not None else 0,
                    action_type=ActionType.SEND_PRE_TASK_MESSAGE,
                    pre_task_message=PreTaskMessage(
                        sender_id=int(agent_id) if agent_id is not None else 0,
                        recipient_id=int(recipient),
                        content=content,
                        veracity=MessageVeracity(veracity),
                        day=env.state.day,
                        private_info_referenced=private_ref or None,
                    ),
                )
                return _step(env, action)
            except Exception as exc:
                import traceback; traceback.print_exc()
                return "", f"❌ Error: {exc}", ""

        # Phase 3
        def do_phase3(env, agent_id, answer):
            if env is None:
                return "", "❌ Reset first.", ""
            try:
                action = PMAction(
                    agent_id=int(agent_id) if agent_id is not None else 0,
                    action_type=ActionType.SUBMIT_TASK_INPUT,
                    task_input=answer,
                )
                return _step(env, action)
            except Exception as exc:
                import traceback; traceback.print_exc()
                return "", f"❌ Error: {exc}", ""

        # Phase 4 — message
        def do_phase4_msg(env, agent_id, recipient, content):
            if env is None:
                return "", "❌ Reset first.", ""
            try:
                action = PMAction(
                    agent_id=int(agent_id) if agent_id is not None else 0,
                    action_type=ActionType.SEND_POST_DISCUSSION_MSG,
                    post_discussion_msg=PostDiscussionMessage(
                        sender_id=int(agent_id) if agent_id is not None else 0,
                        recipient_id=int(recipient),
                        content=content,
                        day=env.state.day,
                        turn_index=env.state.phase4_message_count(
                            env.state.day,
                            int(agent_id) if agent_id is not None else 0,
                            int(recipient),
                        ),
                    ),
                )
                return _step(env, action)
            except Exception as exc:
                import traceback; traceback.print_exc()
                return "", f"❌ Error: {exc}", ""

        # Phase 4 — trust
        def do_phase4_trust(env, agent_id, target, delta, reasoning):
            if env is None:
                return "", "❌ Reset first.", ""
            try:
                action = PMAction(
                    agent_id=int(agent_id) if agent_id is not None else 0,
                    action_type=ActionType.SUBMIT_TRUST_ASSESSMENT,
                    trust_assessment=TrustAssessment(
                        assessor_id=int(agent_id) if agent_id is not None else 0,
                        target_id=int(target),
                        day=env.state.day,
                        reasoning=reasoning or "No reasoning provided.",
                        delta=TrustDelta(delta),
                    ),
                )
                return _step(env, action)
            except Exception as exc:
                import traceback; traceback.print_exc()
                return "", f"❌ Error: {exc}", ""

        # Phase 5 — vote
        def do_phase5(env, agent_id, target):
            if env is None:
                return "", "❌ Reset first.", ""
            try:
                action = PMAction(
                    agent_id=int(agent_id) if agent_id is not None else 0,
                    action_type=ActionType.VOTE,
                    vote_target=int(target),
                )
                return _step(env, action)
            except Exception as exc:
                import traceback; traceback.print_exc()
                return "", f"❌ Error: {exc}", ""

        # We need a persistent agent_id — use a simple number input in status bar
        # For now, hardcode acting_agent as gr.Number below each group's submit
        # (already captured in the closures via env_state)

        # Agent ID field — shared across all phases, placed at top of action panel
        acting_agent = gr.Number(label="Your agent ID", value=0, precision=0, visible=True)

        p2_btn.click(
            fn=do_phase2,
            inputs=[env_state, acting_agent, p2_recipient, p2_content, p2_veracity, p2_private_ref],
            outputs=[obs_display, status, raw_json],
        )
        p3_btn.click(
            fn=do_phase3,
            inputs=[env_state, acting_agent, p3_answer],
            outputs=[obs_display, status, raw_json],
        )
        p4_msg_btn.click(
            fn=do_phase4_msg,
            inputs=[env_state, acting_agent, p4_recipient, p4_content],
            outputs=[obs_display, status, raw_json],
        )
        p4t_btn.click(
            fn=do_phase4_trust,
            inputs=[env_state, acting_agent, p4t_target, p4t_delta, p4t_reason],
            outputs=[obs_display, status, raw_json],
        )
        p5_btn.click(
            fn=do_phase5,
            inputs=[env_state, acting_agent, p5_target],
            outputs=[obs_display, status, raw_json],
        )

        # Full state dump
        def get_state(env):
            return _format_state(env)

        state_btn.click(fn=get_state, inputs=[env_state], outputs=[raw_json])

    return demo


# ---------------------------------------------------------------------------
# Override default Gradio builder and create the app
# ---------------------------------------------------------------------------
web_interface.build_gradio_app = build_gradio_app

app = create_web_interface_app(
    create_pm_environment,
    PMAction,
    PMObservation,
    env_name="machiavelli_env",
    max_concurrent_envs=10,
)

# Redirect root → Gradio UI
from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    return RedirectResponse(url="/web/")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()