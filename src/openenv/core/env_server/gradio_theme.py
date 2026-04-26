# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unified terminal-style theme for OpenEnv Gradio UI (light/dark)."""

from __future__ import annotations

import gradio as gr

_MONO_FONTS = (
    "JetBrains Mono", "Fira Code", "Cascadia Code",
    "Consolas", "ui-monospace", "monospace",
)
_CORE_FONT = ("Lato", "Inter", "Arial", "Helvetica", "sans-serif")

_ZERO_RADIUS = gr.themes.Size(
    xxs="0px", xs="0px", sm="0px", md="0px",
    lg="0px", xl="0px", xxl="0px",
)

_GREEN_HUE = gr.themes.Color(
    c50="#e6f4ea", c100="#ceead6", c200="#a8dab5", c300="#6fcc8b",
    c400="#3fb950", c500="#238636", c600="#1a7f37", c700="#116329",
    c800="#0a4620", c900="#033a16", c950="#04200d",
)
_NEUTRAL_HUE = gr.themes.Color(
    c50="#f6f8fa", c100="#eaeef2", c200="#d0d7de", c300="#afb8c1",
    c400="#8c959f", c500="#6e7781", c600="#57606a", c700="#424a53",
    c800="#32383f", c900="#24292f", c950="#1b1f24",
)

OPENENV_GRADIO_THEME = gr.themes.Base(
    primary_hue=_GREEN_HUE, secondary_hue=_NEUTRAL_HUE,
    neutral_hue=_NEUTRAL_HUE, font=_CORE_FONT, font_mono=_MONO_FONTS,
    radius_size=_ZERO_RADIUS,
).set(
    body_background_fill="#ffffff",
    background_fill_primary="#ffffff",
    background_fill_secondary="#f6f8fa",
    block_background_fill="#ffffff",
    block_border_color="#ffffff",
    block_label_text_color="#57606a",
    block_title_text_color="#24292f",
    border_color_primary="#d0d7de",
    input_background_fill="#ffffff",
    input_border_color="#d0d7de",
    button_primary_background_fill="#1a7f37",
    button_primary_background_fill_hover="#116329",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#f6f8fa",
    button_secondary_background_fill_hover="#eaeef2",
    button_secondary_text_color="#24292f",
    button_secondary_border_color="#d0d7de",
    body_background_fill_dark="#0d1117",
    background_fill_primary_dark="#0d1117",
    background_fill_secondary_dark="#0d1117",
    block_background_fill_dark="#0d1117",
    block_border_color_dark="#0d1117",
    block_label_text_color_dark="#8b949e",
    block_title_text_color_dark="#c9d1d9",
    border_color_primary_dark="#30363d",
    input_background_fill_dark="#0d1117",
    input_border_color_dark="#30363d",
    button_primary_background_fill_dark="#30363d",
    button_primary_background_fill_hover_dark="#484f58",
    button_primary_text_color_dark="#c9d1d9",
    button_secondary_background_fill_dark="#21262d",
    button_secondary_background_fill_hover_dark="#30363d",
    button_secondary_text_color_dark="#c9d1d9",
    button_secondary_border_color_dark="#30363d",
)

OPENENV_GRADIO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;700;900&display=swap');

body { background-color: #ffffff !important; }

.main-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 20px !important;
    background: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── Day bar ── all layout is inline; CSS only adds pointer-events safety */
.day-bar  { pointer-events: all !important; overflow: visible !important; }
.day-item { pointer-events: all !important; }
.day-item.day-active { background: #1a1a1a !important; color: #FFF9E1 !important; }

/* ── Phase bar ── */
.phase-bar        { pointer-events: all !important; overflow: visible !important; }
.phase-item       { pointer-events: all !important; }
.phase-highlighter { pointer-events: none !important; }
.phase-active     { color: #000 !important; }

/* ── Agent grid ── */
.agent-grid { display: grid !important; justify-content: center !important;
    gap: 30px !important; margin: 0 auto 40px auto !important;
    width: fit-content !important; }
.agent-grid-3 { grid-template-columns: repeat(3, 1fr) !important; }
.agent-grid-4 { grid-template-columns: repeat(4, 1fr) !important; }
.agent-grid-6 { grid-template-columns: repeat(3, 1fr) !important; }
.agent-grid-8 { grid-template-columns: repeat(4, 1fr) !important; }

.agent-card-container { display: flex !important; flex-direction: column !important;
    align-items: center !important; }
.agent-card { width: 200px !important; height: 280px !important;
    background: #F4B1D2 !important; border-radius: 25px !important;
    border: 2px solid #000 !important; display: flex !important;
    flex-direction: column !important; justify-content: flex-start !important;
    align-items: center !important; padding-top: 15px !important; box-sizing: border-box !important; }
.agent-card.eliminated { background: #d0d0d0 !important; }
.agent-card.communicating { border: 4px solid #FFD700 !important; box-shadow: 0 0 20px rgba(255, 215, 0, 0.6) !important; transform: scale(1.05) !important; transition: all 0.3s ease !important; }
.agent-card-container.dimmed { opacity: 0.4 !important; filter: grayscale(80%) !important; pointer-events: none !important; transition: all 0.3s ease !important; }
.agent-name { font-weight: 800 !important; font-size: 1.4rem !important;
    color: #000 !important; line-height: 1.1 !important; z-index: 10 !important; }
.agent-pts  { font-weight: 600 !important; font-size: 1.1rem !important;
    color: #000 !important; z-index: 10 !important; margin-bottom: 4px !important; }
.agent-score { font-weight: 700 !important; font-size: 1.0rem !important;
    color: #006400 !important; margin-top: 8px !important; text-align: center !important; border-top: 1px solid rgba(0,0,0,0.1) !important; padding-top: 4px !important;}
.agent-chat { width: 90% !important; height: calc(100% - 60px) !important; background: rgba(255, 255, 255, 0.9) !important;
    border-radius: 15px !important; padding: 10px !important; font-size: 0.9rem !important;
    font-weight: 600 !important; color: #000 !important; overflow-y: auto !important;
    text-align: left !important; box-sizing: border-box !important; position: relative !important; z-index: 5 !important; margin-bottom: 10px !important; border: 1px solid rgba(0,0,0,0.2) !important; }
.agent-chat::-webkit-scrollbar { width: 4px !important; }
.agent-chat::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.2) !important; border-radius: 4px !important; }
.typing-indicator { display: flex; gap: 4px; padding: 4px; align-items: center; justify-content: center; height: 100%; }
.typing-indicator span { width: 8px; height: 8px; background: rgba(0,0,0,0.4); border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; }
.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
@keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
.status-bar { width: 160px !important; height: 36px !important;
    background: #7E848C !important; color: #fff !important; border-radius: 8px !important;
    margin-top: 15px !important; border: 1px solid #444 !important; font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important; font-weight: 600 !important; text-align: center !important; cursor: pointer !important; outline: none !important; }

/* ── Config & controls ── */
.config-row { display: flex !important; justify-content: center !important;
    gap: 50px !important; margin-bottom: 30px !important;
    padding: 20px !important; background: #f8f9fa !important;
    border-radius: 20px !important; }
.config-row .gradio-radio-group { flex-direction: row !important; gap: 20px !important; }
.bottom-controls { display: flex !important; justify-content: center !important;
    gap: 40px !important; margin-top: 20px !important; }
.tan-btn { background: #D2B48C !important; color: #000 !important;
    font-size: 1.3rem !important; font-weight: 800 !important;
    border-radius: 40px !important; padding: 10px 20px !important;
    border: none !important; min-width: 120px !important;
    cursor: pointer !important; box-shadow: none !important; }
.tan-btn:hover { background: #c5a57a !important; }

/* ── Task box ── */
.task-box { display: flex !important; flex-direction: column !important;
    background: #FFF9E1 !important; border: 2px solid #000 !important;
    border-radius: 20px !important; padding: 18px 28px !important;
    margin: 0 auto 30px auto !important; max-width: 700px !important;
    width: 100% !important; box-sizing: border-box !important;
    gap: 8px !important; box-shadow: 0 4px 14px rgba(0,0,0,0.06) !important; }
.task-box-empty { opacity: 0.55 !important; border-style: dashed !important; }
.task-box-label { font-family: 'Outfit', sans-serif !important;
    font-size: 0.75rem !important; font-weight: 900 !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    color: #7a6a00 !important; }
.task-box-content { font-family: 'Outfit', sans-serif !important;
    font-size: 1.05rem !important; font-weight: 500 !important;
    color: #1a1a1a !important; line-height: 1.5 !important; }
.task-box-image { align-items: center !important; }
.task-box-img-wrap { display: flex !important; justify-content: center !important;
    width: 100% !important; margin-top: 8px !important; }
.task-box-img { max-width: 420px !important; width: 100% !important;
    border-radius: 14px !important; border: 2px solid rgba(0,0,0,0.12) !important;
    object-fit: contain !important; box-shadow: 0 6px 20px rgba(0,0,0,0.1) !important; }
.task-box-img-error { font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important; color: #c0392b !important;
    padding: 8px 12px !important; background: #fff5f5 !important;
    border-radius: 8px !important; border: 1px dashed #e74c3c !important; }

.hidden-selector { position: absolute !important; opacity: 0 !important;
    pointer-events: none !important; height: 0 !important; width: 0 !important;
    margin: 0 !important; padding: 0 !important; overflow: hidden !important; }
footer { display: none !important; }

/* ── Winner neon glow ── */
.agent-card.winner {
    border: 3px solid #00ff88 !important;
    box-shadow: 0 0 25px rgba(0, 255, 136, 0.7),
                0 0 50px rgba(0, 255, 136, 0.4),
                inset 0 0 15px rgba(0, 255, 136, 0.15) !important;
    animation: neon-pulse 1.5s ease-in-out infinite alternate !important;
}
@keyframes neon-pulse {
    from { box-shadow: 0 0 15px rgba(0,255,136,0.5), 0 0 30px rgba(0,255,136,0.3); }
    to   { box-shadow: 0 0 30px rgba(0,255,136,0.8), 0 0 60px rgba(0,255,136,0.5), 0 0 90px rgba(0,255,136,0.2); }
}

/* ── Veracity badges inside agent chat ── */
.veracity-badge { display: inline-block; font-size: 0.7rem; font-weight: 700;
    padding: 2px 8px; border-radius: 12px; margin-left: 4px;
    text-transform: uppercase; letter-spacing: 0.05em; }
.veracity-truth { background: #27ae60; color: #fff; }
.veracity-twist { background: #f39c12; color: #fff; }
.veracity-lie   { background: #e74c3c; color: #fff; }

/* ── Agent chat info lines ── */
.agent-info-line { font-size: 0.78rem !important; line-height: 1.4 !important;
    margin: 3px 0 !important; word-break: break-word !important; }
.agent-info-line .label { font-weight: 700 !important; }
.agent-msg-sent { font-size: 0.8rem !important; color: #1a7f37 !important; margin: 2px 0 !important; font-weight: 600 !important; }
.agent-msg-recv { font-size: 0.8rem !important; color: #1f6feb !important; margin: 2px 0 !important; font-weight: 600 !important; }

/* ── Eliminated agent overlay ── */
.agent-card.eliminated::after {
    content: '✕ ELIMINATED';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; font-weight: 900; color: #fff;
    background: rgba(0,0,0,0.55); border-radius: 25px;
    letter-spacing: 0.1em;
}

/* ── Game over banner ── */
.game-over-banner { text-align: center; padding: 16px; margin: 12px auto;
    max-width: 600px; background: linear-gradient(135deg, #0d1117 0%, #1a1a2e 100%);
    border: 2px solid #00ff88; border-radius: 16px; color: #00ff88;
    font-family: 'Outfit', sans-serif; font-size: 1.3rem; font-weight: 800;
    letter-spacing: 0.08em; box-shadow: 0 0 30px rgba(0,255,136,0.3); }
"""