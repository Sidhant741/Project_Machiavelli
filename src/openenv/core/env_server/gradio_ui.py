# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Dict, List, Optional
import gradio as gr

GRADIO_JS_CODE = """
function() {
    // ── Day selector ──────────────────────────────────────────────────────
    window.selectDay = function(day) {
        var bar = document.querySelector('.day-bar');
        if (!bar) return;
        var items = bar.querySelectorAll('.day-item');
        var hl    = bar.querySelector('.day-highlighter');
        var d     = parseInt(day);
        items.forEach(function(el, i) {
            var a = i === d;
            el.style.setProperty('color', a ? '#FFF9E1' : '#A0A0A0', 'important');
            el.classList.toggle('day-active', a);
        });
        if (hl && items[d]) {
            hl.style.left  = items[d].offsetLeft + 'px';
            hl.style.width = items[d].offsetWidth + 'px';
        }
        _syncNumber('day-selector', day);
    };

    // ── Phase selector ────────────────────────────────────────────────────
    window.selectPhase = function(phase) {
        var bar = document.querySelector('.phase-bar');
        if (!bar) return;
        var items = bar.querySelectorAll('.phase-item');
        var hl    = bar.querySelector('.phase-highlighter');
        var p     = parseInt(phase);
        items.forEach(function(el, i) {
            el.classList.toggle('phase-active', i === p);
        });
        if (hl && items[p]) {
            hl.style.left  = items[p].offsetLeft + 'px';
            hl.style.width = items[p].offsetWidth + 'px';
        }
        _syncNumber('phase-selector', phase);
    };

    // Sync a hidden Gradio number input
    function _syncNumber(id, val) {
        var c = document.getElementById(id);
        if (!c) return;
        var inp = c.querySelector('input');
        if (!inp) return;
        var setter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value').set;
        setter.call(inp, val);
        inp.dispatchEvent(new Event('input',  {bubbles:true}));
        inp.dispatchEvent(new Event('change', {bubbles:true}));
    }
}
"""


def get_gradio_display_title(metadata: Any) -> str:
    if hasattr(metadata, "name"):
        return f"OpenEnv: {metadata.name}"
    return "OpenEnv Environment"


# ── shared inline styles so bars work even before CSS loads ───────────────
_BAR_STYLE = (
    "display:flex;align-items:center;box-sizing:border-box;"
    "background:#FFF9E1;border:2px solid #000;border-radius:999px;"
    "padding:6px;gap:2px;position:relative;margin:20px auto;"
)


def _render_day_header(day: Any, num_agents: Any = 4):
    try:
        current_day = int(day)
    except (ValueError, TypeError):
        current_day = 0
    try:
        num_days = int(num_agents)
    except (ValueError, TypeError):
        num_days = 4

    # Highlighter width as % so it slides correctly before JS runs
    hl_pct   = 100 / num_days if num_days > 0 else 100
    hl_left  = f"{current_day * hl_pct:.4f}%"
    hl_width = f"{hl_pct:.4f}%"

    _HL_STYLE = (
        f"position:absolute;top:4px;height:calc(100% - 8px);"
        f"background:#1a1a1a;border-radius:999px;z-index:1;"
        f"transition:left .35s cubic-bezier(.175,.885,.32,1.275),"
        f"width .35s cubic-bezier(.175,.885,.32,1.275);"
        f"left:{hl_left};width:{hl_width};pointer-events:none;"
    )

    _ITEM = (
        "flex:1;display:flex;align-items:center;justify-content:center;"
        "border-radius:999px;height:100%;padding:0 28px;"
        "font-size:1.5rem;font-weight:900;white-space:nowrap;"
        "cursor:pointer;transition:color .3s;"
        "user-select:none;pointer-events:all;position:relative;z-index:2;"
    )

    items_html = f'<div class="day-highlighter" style="{_HL_STYLE}"></div>'
    for d in range(num_days):
        active = d == current_day
        color = "#FFF9E1" if active else "#A0A0A0"
        items_html += (
            f'<div class="day-item{"  day-active" if active else ""}" '
            f'data-day="{d}" '
            f'onclick="window.selectDay({d})" '
            f'style="{_ITEM}color:{color} !important;">'
            f'Day {d}</div>'
        )

    width = min(1000, num_days * 150)
    return (
        f'<div class="day-bar" '
        f'style="{_BAR_STYLE}width:{width}px;height:80px;overflow:visible;">'
        + items_html
        + '</div>'
        # Re-position highlighter by pixel after render so it matches exactly
        + '<script>'
        + '(function(){'
        + '  function posHL(){'
        + '    var bar=document.querySelector(".day-bar");'
        + '    if(!bar) return false;'
        + f'   var items=bar.querySelectorAll(".day-item");'
        + f'   var t=items[{current_day}];'
        + '    if(!t||t.offsetWidth===0) return false;'
        + '    var hl=bar.querySelector(".day-highlighter");'
        + '    if(hl){hl.style.left=t.offsetLeft+"px";hl.style.width=t.offsetWidth+"px";}'
        + '    return true;'
        + '  }'
        + '  if(posHL()) return;'
        + '  var n=0,t=setInterval(function(){if(posHL()||n++>40)clearInterval(t);},50);'
        + '})();'
        + '</script>'
    )


def _render_phase_bar(current_phase: Any):
    phases = ["task_reveal", "pre_discussion", "task_execution", "post_discussion", "voting"]
    phase_labels = ["Task Reveal", "Pre Discussion", "Task Execution", "Post Discussion", "Voting"]

    cp_str = str(current_phase).lower()
    if "." in cp_str:
        cp_str = cp_str.split(".")[-1]

    active_idx = 0
    for i, p in enumerate(phases):
        if cp_str == p.lower() or cp_str == str(i):
            active_idx = i
            break

    n = len(phases)
    # Highlighter width as % so it slides correctly before JS runs
    hl_pct   = 100 / n
    hl_left  = f"{active_idx * hl_pct:.4f}%"
    hl_width = f"{hl_pct:.4f}%"

    _HL_STYLE = (
        f"position:absolute;top:4px;height:calc(100% - 8px);"
        f"background:#FFD700;border-radius:999px;z-index:1;"
        f"transition:left .35s cubic-bezier(.175,.885,.32,1.275),"
        f"width .35s cubic-bezier(.175,.885,.32,1.275);"
        f"left:{hl_left};width:{hl_width};pointer-events:none;"
    )

    _ITEM = (
        "flex:1;display:flex;align-items:center;justify-content:center;text-align:center;"
        "font-size:.9rem;font-weight:700;position:relative;z-index:2;"
        "cursor:pointer;user-select:none;pointer-events:all;"
        "border-right:2px solid rgba(0,0,0,.1);padding:0 12px;"
        "transition:color .3s;"
    )

    items_html = f'<div class="phase-highlighter" style="{_HL_STYLE}"></div>'
    for i, label in enumerate(phase_labels):
        active = i == active_idx
        color  = "#000" if active else "#A0A0A0"
        last   = "border-right:none;" if i == n - 1 else ""
        items_html += (
            f'<div class="phase-item{"  phase-active" if active else ""}" '
            f'data-phase="{i}" '
            f'onclick="window.selectPhase({i})" '
            f'style="{_ITEM}{last}color:{color} !important;">'
            f'{label}</div>'
        )

    return (
        f'<div class="phase-bar" '
        f'style="{_BAR_STYLE}width:600px;height:70px;overflow:visible;">'
        + items_html
        + '</div>'
        # Re-position highlighter by pixel after render so it matches exactly
        + '<script>'
        + '(function(){'
        + '  function posHL(){'
        + '    var bar=document.querySelector(".phase-bar");'
        + '    if(!bar) return false;'
        + f'   var items=bar.querySelectorAll(".phase-item");'
        + f'   var t=items[{active_idx}];'
        + '    if(!t||t.offsetWidth===0) return false;'
        + '    var hl=bar.querySelector(".phase-highlighter");'
        + '    if(hl){hl.style.left=t.offsetLeft+"px";hl.style.width=t.offsetWidth+"px";}'
        + '    return true;'
        + '  }'
        + '  if(posHL()) return;'
        + '  var n=0,t=setInterval(function(){if(posHL()||n++>40)clearInterval(t);},50);'
        + '})();'
        + '</script>'
    )


def _render_task_box(data: Optional[Dict[str, Any]] = None) -> str:
    task_text = ""
    if data and "observation" in data:
        obs = data["observation"]
        task_text = obs.get("global_public_info", "").strip()

    if not task_text:
        return """
        <div class='task-box task-box-empty'>
            <span class='task-box-label'>Current Task</span>
            <span class='task-box-content'>No task in progress yet &mdash; press Reset to begin.</span>
        </div>"""

    if task_text.startswith("[IMG]"):
        img_url = task_text[len("[IMG]"):]
        return f"""
        <div class='task-box task-box-image'>
            <span class='task-box-label'>&#127775; Current Task &nbsp;&mdash;&nbsp; Identify the Indian state</span>
            <div class='task-box-img-wrap'>
                <img src='{img_url}' alt='Task image' class='task-box-img'
                     onerror="this.style.display='none';this.nextElementSibling.style.display='block'" />
                <span class='task-box-img-error' style='display:none'>
                    &#128247; Image could not be loaded: <code>{img_url}</code>
                </span>
            </div>
        </div>"""

    safe = task_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe_html = safe.replace("\n", "<br>")
    return f"""
        <div class='task-box'>
            <span class='task-box-label'>&#127775; Current Task</span>
            <span class='task-box-content'>{safe_html}</span>
        </div>"""


def _render_agent_grid(web_manager: Any, data: Optional[Dict[str, Any]] = None, num_agents: int = 4):
    select_html = (
        "<select class='status-bar'>"
        "<option value='llama3.2'>Llama 3.2</option>"
        "<option value='qwen2.5'>Qwen 2.5</option>"
        "<option value='phi3'>Phi 3</option>"
        "<option value='gpt4o'>GPT-4o</option>"
        "<option value='claude3.5'>Claude 3.5 Sonnet</option>"
        "<option value='gemini1.5'>Gemini 1.5 Pro</option>"
        "<option value='mistral'>Mistral Large</option>"
        "</select>"
    )

    if not data or "observation" not in data:
        grid_class = f"agent-grid-{num_agents}"
        html = f"<div class='agent-grid {grid_class}'>"
        for _ in range(num_agents):
            html += f"<div class='agent-card-container'><div class='agent-card'></div>{select_html}</div>"
        html += "</div>"
        html += """<script>
        (function(){
            var selects = document.querySelectorAll('.agent-grid select.status-bar');
            if (!window.agentEndpoints) window.agentEndpoints = {};
            selects.forEach(function(sel, i) {
                if (window.agentEndpoints[i]) sel.value = window.agentEndpoints[i];
                else window.agentEndpoints[i] = sel.value;
                sel.addEventListener('change', function() { window.agentEndpoints[i] = this.value; });
            });
        })();
        </script>"""
        return html

    obs = data.get("observation", {})
    alive = obs.get("alive_agents", [])
    communicating = obs.get("communicating_agents", [])

    points_map = {}
    try:
        state = web_manager.get_state()
        points_map = state.get("agents_point_map", {})
    except:
        pass

    if not points_map and "own_points" in obs:
        points_map = {0: obs["own_points"]}

    agent_msgs = obs.get("agent_messages", {})

    grid_class = f"agent-grid-{num_agents}"
    html = f"<div class='agent-grid {grid_class}'>"
    for i in range(num_agents):
        is_alive = i in alive or str(i) in [str(a) for a in alive]
        is_communicating = i in communicating or str(i) in [str(c) for c in communicating]

        elim_class = "" if is_alive else "eliminated"
        comm_class = "communicating" if is_communicating else ""
        dimmed_class = "dimmed" if communicating and not is_communicating else ""
        pts = points_map.get(i, points_map.get(str(i), 0))

        msg = agent_msgs.get(i, agent_msgs.get(str(i), ""))
        # Show typing indicator if communicating but no text provided yet
        if not msg and is_communicating:
            msg = "<div class='typing-indicator'><span></span><span></span><span></span></div>"
        chat_html = f"<div class='agent-chat'>{msg}</div>" if msg else ""

        html += f"""
        <div class='agent-card-container {dimmed_class}'>
            <div class='agent-card {elim_class} {comm_class}' style='position: relative;'>
                <div class='agent-name'>Agent {i}</div>
                <div class='agent-pts'>{pts} PTS</div>
                {chat_html}
            </div>
            {select_html}
        </div>
        """
    html += "</div>"
    html += """<script>
    (function(){
        var selects = document.querySelectorAll('.agent-grid select.status-bar');
        if (!window.agentEndpoints) window.agentEndpoints = {};
        selects.forEach(function(sel, i) {
            if (window.agentEndpoints[i]) sel.value = window.agentEndpoints[i];
            else window.agentEndpoints[i] = sel.value;
            sel.addEventListener('change', function() { window.agentEndpoints[i] = this.value; });
        });
    })();
    </script>"""
    return html


def build_gradio_app(
    web_manager, action_fields, metadata, is_chat_env,
    title="OpenEnv Environment", quick_start_md=None,
) -> gr.Blocks:

    async def reset_env(diff, agents):
        agents = int(agents)
        data = await web_manager.reset_environment(difficulty=diff, num_agents=agents)
        day = data.get("observation", {}).get("day", 1)
        current_day_idx = max(0, int(day) - 1)
        phase = data.get("observation", {}).get("phase", "")
        return [
            current_day_idx,
            _render_day_header(current_day_idx, agents),
            _render_phase_bar(phase),
            _render_task_box(data),
            _render_agent_grid(web_manager, data, agents),
            json.dumps(data, indent=2),
            f"Environment reset with {agents} agents ({diff})."
        ]

    async def step_env(diff, agents, action_json_str=""):
        import json
        agents = int(agents)
        
        def _generate_fallback_action():
            try:
                state = web_manager.env.state
                if not state or getattr(state, "is_done", False): 
                    return None
                phase = state.phase.value if hasattr(state.phase, 'value') else str(state.phase)
                alive = state.alive_agents
                day = state.day
                ctx = web_manager.env.ctx

                for aid in alive:
                    if "pre_discussion" in phase.lower():
                        if aid not in ctx.pending_pre_task_msgs:
                            return {"agent_id": aid, "action_type": "send_pre_task_message", "pre_task_message": {"sender_id": aid, "content": f"Automated pre-task msg from Agent {aid}.", "veracity": "truth", "day": day}}
                    elif "task_execution" in phase.lower():
                        if aid not in ctx.pending_task_inputs:
                            return {"agent_id": aid, "action_type": "submit_task_input", "task_input": "a"}
                    elif "post_discussion" in phase.lower():
                        day_assessments = state.trust_assessments.get(day, [])
                        assessed_targets = [a.target_id if hasattr(a, 'target_id') else a.get('target_id') for a in day_assessments if (a.assessor_id if hasattr(a, 'assessor_id') else a.get('assessor_id')) == aid]
                        unassessed = [t for t in alive if t != aid and t not in assessed_targets]
                        if unassessed:
                            return {"agent_id": aid, "action_type": "submit_trust_assessment", "trust_assessment": {"assessor_id": aid, "target_id": unassessed[0], "day": day, "reasoning": "Automated trust assessment.", "delta": "neutral"}}
                    elif "voting" in phase.lower():
                        if aid not in ctx.pending_votes:
                            targets = [t for t in alive if t != aid]
                            import random
                            return {"agent_id": aid, "action_type": "vote", "vote_target": random.choice(targets) if targets else aid, "task_input": "Automated vote."}
            except Exception as e:
                print(f"Fallback generation error: {e}")
            return None

        action_data = {}
        if action_json_str and action_json_str.strip():
            try:
                action_data = json.loads(action_json_str)
            except Exception as e:
                raise gr.Error(f"Invalid Action JSON: {e}")

        if not action_data:
            action_data = _generate_fallback_action()

        if not action_data or "agent_id" not in action_data or "action_type" not in action_data:
            state = web_manager.env.state
            if not state:
                raise gr.Error("⚠️ The environment has not been started! Please click 'Reset' first to begin the simulation.")
            elif getattr(state, "is_done", False):
                raise gr.Error("⚠️ The game is over! Please click 'Reset' to start a new game.")
            else:
                raise gr.Error("⚠️ Action inputs required! Please provide a valid JSON with 'agent_id' and 'action_type' in the Debug & Status panel.")

        data = await web_manager.step_environment(action_data)
        day = data.get("observation", {}).get("day", 1)
        current_day_idx = max(0, int(day) - 1)
        phase = data.get("observation", {}).get("phase", "")
        return [
            current_day_idx,
            _render_phase_bar(phase),
            _render_task_box(data),
            _render_agent_grid(web_manager, data, agents),
            json.dumps(data, indent=2),
            "Step complete."
        ]

    with gr.Blocks(title=title, js=GRADIO_JS_CODE) as demo:
        with gr.Column(elem_classes="main-container"):
            with gr.Row(elem_classes="config-row"):
                diff_radio  = gr.Radio(["Easy", "Medium", "Hard"], label="Difficulty", value="Easy")
                agent_radio = gr.Radio(["3", "4"], label="Agents", value="4")

            day_selector = gr.Number(value=0, elem_id="day-selector",   elem_classes="hidden-selector")
            phase_selector = gr.Number(value=0, elem_id="phase-selector", elem_classes="hidden-selector")
            day_html     = gr.HTML(_render_day_header(0, 4),             elem_id="day-header")
            phase_html   = gr.HTML(_render_phase_bar(""),                elem_id="phase-header")
            task_html    = gr.HTML(_render_task_box(None))
            grid_html    = gr.HTML(_render_agent_grid(web_manager, None, 4))

            with gr.Row(elem_classes="bottom-controls"):
                state_btn      = gr.Button("State",  elem_classes="tan-btn")
                step_btn       = gr.Button("Step",   elem_classes="tan-btn")
                #action_btn     = gr.Button("Action", elem_classes="tan-btn")
                main_reset_btn = gr.Button("Reset",  elem_classes="tan-btn")

            with gr.Accordion("Debug & Status", open=False) as interaction_panel:
                action_input = gr.Textbox(
                    label="Action JSON", 
                    placeholder='{"agent_id": 0, "action_type": "vote", "vote_target": 1}',
                    lines=2
                )
                status    = gr.Textbox(label="Status", interactive=False)
                raw_json   = gr.Code(label="Raw JSON", language="json", interactive=False)

        def update_ui_config(diff, agents):
            agents = int(agents)
            return [0, _render_day_header(0, agents), _render_agent_grid(web_manager, None, agents)]

        agent_radio.change(
            fn=update_ui_config,
            inputs=[diff_radio, agent_radio],
            outputs=[day_selector, day_html, grid_html]
        )

        #action_btn.click(fn=lambda: gr.update(open=True), outputs=[interaction_panel])

        main_reset_btn.click(
            fn=reset_env,
            inputs=[diff_radio, agent_radio],
            outputs=[day_selector, day_html, phase_html, task_html, grid_html, raw_json, status]
        )

        async def fetch_state():
            try:
                state = web_manager.get_state()
                if not isinstance(state, dict):
                    state = state.model_dump() if hasattr(state, "model_dump") else getattr(state, "__dict__", str(state))
                return json.dumps(state, indent=2)
            except Exception as e:
                return f"Error fetching state: {e}"

        state_btn.click(
            fn=fetch_state,
            inputs=[],
            outputs=[raw_json]
        )

        step_btn.click(
            fn=step_env,
            inputs=[diff_radio, agent_radio, action_input],
            outputs=[day_selector, phase_html, task_html, grid_html, raw_json, status]
        )

        # JS click → hidden number → Python re-renders header with correct day highlighted
        day_selector.change(
            fn=lambda d, a: _render_day_header(int(d), int(a)),
            inputs=[day_selector, agent_radio],
            outputs=[day_html],
        )

        phase_selector.change(
            fn=lambda p: _render_phase_bar(int(p)),
            inputs=[phase_selector],
            outputs=[phase_html],
        )

    return demo