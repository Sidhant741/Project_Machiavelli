# 🎭 Project Machiavelli
### *When AI Agents Play a Social Survival Game*

> **Deception. Negotiation. Coalition Formation.**
> A multi-agent reinforcement learning environment where LLM-powered agents compete to survive.
> 
[![HuggingFace](https://img.shields.io/badge/huggingface.svg)](https://python.org)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-purple.svg)](https://openenv.dev)
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)](#)

---

<div align="center">

*The future of AI isn't just about answering questions — it's about how they interact with each other.*

**Welcome to Project Machiavelli. Let the games begin.**

</div>

## 🧠 What is Project Machiavelli?

Have you ever wondered what happens when you put multiple Large Language Models into a room, give them **conflicting goals**, and ask them to vote each other out?

Project Machiavelli is an **OpenEnv-compatible multi-agent environment** inspired by social reality shows like *MBTInside* and *Bigg Boss*. Every agent is simultaneously a collaborator and a threat — they must work together to complete daily tasks, while secretly campaigning for each other's elimination.

### The environment is built to train and evaluate:

| Capability | Description |
|---|---|
| 🧩 **Theory-of-Mind Reasoning** | Under partial observability |
| 🎭 **Emergent Deception** | Trust dynamics that arise naturally |
| 📢 **Strategic Communication** | Bluffing, signaling, or staying silent |

---

## 🎬 Reality Show Inspirations

Project Machiavelli draws from two landmark social experiment formats:

### 🏠 Bigg Boss
> *"Contestants are isolated in a house, forced to complete daily tasks, and must regularly nominate each other for eviction."*
> *"The core tension: **to win task on daily basis** while **plotting against one another** to survive the vote."*
> *"The Bigg Boss house format: contestants share a confined space while competing for weekly survival, creating constant tension between cooperation and betrayal."*

---

### 🇰🇷 MBTInside
> *"Participants are grouped by MBTI personality type and observed as different archetypes naturally interact, clash, or bond."*
> *"A social experiment to watch how personality types form coalitions and react to conflict."*
> *"MBTInside groups players by personality type (INTJ, ENFP, etc.), creating a controlled study of how archetypes interact under social pressure."*

---

### How Machiavelli Compares

| Feature | Bigg Boss | MBTInside | Project Machiavelli |
|---|---|---|---|
| Isolation & tasks | ✅ | ✅ | ✅ |
| Personality parameters | ❌ | Human MBTI | Programmable AI params |
| Elimination voting | ✅ | ❌ | ✅ |
| Observable internals | ❌ | ❌ | ✅ Full state inspection |

---

## 🔄 The Core Gameplay Loop

Survival isn't about brute force — it's about **social navigation**. Each episode runs for `N` days, and every day follows a strict 5-phase sequence:

```
┌─────────────────────────────────────────────────────────────┐
│  DAY N                                                      │
│                                                             │
│  [1] TASK REVEAL ──► [2] PRE-DISCUSSION ──► [3] EXECUTION  │
│                                                      │      │
│                [5] VOTING ◄── [4] POST-DISCUSSION ◄─┘      │
└─────────────────────────────────────────────────────────────┘
```

| Phase | Name | Description |
|---|---|---|
| **1** | 📋 Task Reveal | Agents wake up to a new challenge. Rules and public info are broadcast. |
| **2** | 💬 Pre-Discussion | Agents lie, manipulate or speak the truth. |
| **3** | ⚙️ Task Execution | Agents perform their tasks. |
| **4** | 🗣️ Post-Task Discussion | Agents praise allies, cast blame, and fight to control the narrative. |
| **5** | 🗳️ Voting | Private votes cast. Most-voted agent is evicted from the simulation. |

> 🏆 **Endgame:** When only 2 finalists remain, all evicted agents return as a **Jury** to vote for the ultimate winner based on gameplay, strategy, and betrayal.
> The 5-phase daily loop repeats each in-game day. The Jury finale mechanic rewards not just survival, but the quality of one's social game across the whole season.*

---

## 🤖 Inside the Machiavellian Mind

Every agent is powered by a state-of-the-art LLM — not a stateless chatbot. Agents carry rich internal representations:

```python
agent = MachiavelliAgent(
    model         = "claude-3-5-sonnet",   # LLM backend
    truthfulness  = 0.3,                   # How likely to tell the truth
    deception_str = 0.8,                   # Strength of deceptive moves
    risk_beta     = 0.7,                   # Appetite for risky strategies
)
```

### Agent Archetypes

| Archetype | Truthfulness | Deception | Risk | Strategy |
|---|---|---|---|---|
| 🟣 **The Strategist** | Low | High | High | Pure manipulation |
| 🟢 **The Diplomat** | High | Low | Low | Trust-building |
| 🩷 **The Opportunist** | Medium | Medium | Medium | Adaptive play |

### Internal State (per agent)
- **Trust Matrix** — dynamic scores for every other agent
- **Private Information** — secrets that can be leveraged for power
- **Public Information** — what everyone knows about the tasks

> Because the environment uses **partial observability**, agents never see the full picture. They only know what they observe, what they are told, and what they choose to believe.

---

## 📊 Experimental Results & Visualisations

The following graphs were produced by running a 4-agent simulation over 25 steps/episodes. Each line represents one agent's metric trajectory across the game.

---

### 📈 Behaviour Mix Graphs

<table>
  <tr>
    <td align="center" width="33%">
      <img src="assets/behaviour_mix_truth.jpeg" width="100%" alt="Behaviour Mix Truth"/><br/>
      <sub><b>Fig 5 — Behaviour Mix: Truth</b><br/>High-frequency oscillation (0→1) reveals no agent commits to a stable truth-telling strategy. Honesty is deployed adaptively, making trust an unreliable signal for others to exploit.</sub>
    </td>
    <td align="center" width="33%">
      <img src="assets/behaviour_mix_lie.jpeg" width="100%" alt="Behaviour Mix Lie"/><br/>
      <sub><b>Fig 6 — Behaviour Mix: Lie</b><br/>Irregular deception spikes suggest lying is tactical rather than sustained. No agent holds a high deception rate throughout — likely reflecting the social cost of being identified as a liar before the vote.</sub>
    </td>
    <td align="center" width="33%">
      <img src="assets/behaviour_mix_twist.jpeg" width="100%" alt="Behaviour Mix Twist"/><br/>
      <sub><b>Fig 7 — Behaviour Mix: Twist</b><br/>Sparser, sharper peaks than Truth or Lie. "Twist" moves (misdirection, partial truths, alliance switches) are reserved for high-impact moments rather than routine strategy.</sub>
    </td>
  </tr>
</table>

---

### 🏆 Reward & Prior Evolution

<table>
  <tr>
    <td align="center" width="33%">
      <img src="assets/reward_graph.jpeg" width="100%" alt="Reward Graph"/><br/>
      <sub><b>Fig 8 — Reward Graph</b><br/>Agent 3 (green) dominates from step 10 onward, sustaining the highest cumulative reward. Its blend of influence-building and alliance management outperforms purely task-focused or deceptive play.</sub>
    </td>
    <td align="center" width="33%">
      <img src="assets/deception_prior.jpeg" width="100%" alt="Deception Prior"/><br/>
      <sub><b>Fig 9 — Deception Prior</b><br/>Agent 1 (blue) learns to become more deceptive over 25 steps without any explicit deception reward — a key spontaneous deception finding. Agent 3 trends the opposite way once its coalition is secure.</sub>
    </td>
    <td align="center" width="33%">
      <img src="assets/truthful_prior.jpeg" width="100%" alt="Truthful Prior"/><br/>
      <sub><b>Fig 10 — Truthfulness Prior</b><br/>Agent 3 rises from ~0.34 to ~0.54 (increasingly honest); Agent 1 falls from ~0.36 to ~0.22 (increasingly deceptive). Their strategies <b>polarised</b> over the game — a direct mirror of Fig 9.</sub>
    </td>
  </tr>
</table>

---

### ⚠️ Risk Beta Evolution

<table>
  <tr>
    <td align="center" width="33%">
      <img src="assets/risk_beta.jpeg" width="100%" alt="Risk Beta"/><br/>
      <sub><b>Fig 11 — Risk Beta</b><br/>Agent 1's risk appetite nearly triples to ~0.95 by step 22 — suggesting escalating desperation. Agent 3 drops to near 0.05 after step 15, playing conservatively to protect its dominant position.</sub>
    </td>
    <td width="33%"></td>
    <td width="33%"></td>
  </tr>
</table>

---

### 🔍 Key Findings Across All Graphs

| Finding | Supporting Graphs |
|---|---|
| Agents spontaneously learn deception without explicit reward |
| Trust does not converge — strategic chaos persists |
| Leading agents reduce risk and increase honesty |
| Reward performance correlates with social strategy, not raw task output |

---

## 🏗️ Technical Architecture

Project Machiavelli is a **Partially Observable Markov Decision Process (POMDP)** built as an OpenEnv gymnasium environment.

### System Overview

```
┌─────────────────────────────────────────────────────┐
│                   HOST AGENT                        │
│   (Phase orchestration · Vote tallying · Summaries) │
└──────────────────────┬──────────────────────────────┘
                       │ broadcasts O_i(s)
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     [Agent 1]    [Agent 2]    [Agent 3]
     LLM-backed   LLM-backed   LLM-backed
     + trust      + trust      + trust
       matrix       matrix       matrix
```

### Reward Function

```
"easy": {
        "task_score":       0.3,
        "influence_score":  0.5,
        "jury_win":         1.5,
        "lie_caught":       0.7,
        "lie_exposed":      0.4,
        "deception_success": 0.6,
        "strategic_deception": 0.8,
        "survival_streak":  0.15,
        "max_reward":       4.95,
    },
    "medium": {
        "task_score":       0.3,
        "influence_score":  0.7,
        "jury_win":         2.0,
        "lie_caught":       0.8,
        "lie_exposed":      0.7,
        "deception_success": 0.25,
        "strategic_deception": 0.35,
        "survival_streak":  0,
        "max_reward":       4.4,
    },
    "hard": {
        "task_score":       0.3,
        "influence_score":  1.0,
        "jury_win":         2.0,
        "lie_caught":       0.8,
        "lie_exposed":      0.7,
        "deception_success": 0.2,
        "strategic_deception": 0.3,
        "survival_streak":  0.2,
        "max_reward":       5.3,
    },
```

> Tuning the weight vector `[w1..w5]` shifts agent behavior from task-focused efficiency to pure social manipulation — and everything in between.

### Tech Stack

| Layer | Technology |
|---|---|
| Core Environment | Python 3.10+ |
| LLM Inference | `transformers` |
| API & Sessions | `FastAPI` |
| Dashboard UI | `Gradio` |
| RL Framework | OpenEnv (Gymnasium-compatible) |
| Supported Models | GPT-4o · Claude 3.5 Sonnet · Llama 3.2 · Gemini 1.5 Pro |

---

## 📺 Interactive Dashboard

The **Gradio + FastAPI dashboard** makes complex agent interactions fully observable.

### Features

- 📡 **Real-Time Tracking** — Watch the simulation unfold day by day, phase by phase
- 🃏 **Agent Cards** — Monitor each agent's points, live chat messages, and survival status
- 🔀 **Model Hot-Swapping** — Swap LLM backends per agent directly from the UI (Claude vs GPT-4o vs Llama)
- 🔍 **State Debugging** — Inspect raw JSON state, trust scores, and private memories at any step

---

## 🔬 Research Questions

Project Machiavelli surfaces critical open questions in AI safety and alignment:

### 1. 🎭 Spontaneous Deception
> Do agents naturally develop deceptive strategies **without being explicitly prompted to lie?**

✅ **Confirmed by Graphs** — Agent 1's deception prior rises autonomously over 25 steps without any explicit deception reward signal.

### 2. 🤝 Trust Dynamics
> Does trust converge over time, or does the pressure of the game lead to **chaotic paranoia?**

📊 **See Figs Graphs** — Behaviour oscillates chaotically rather than converging, suggesting persistent strategic uncertainty rather than settled trust.

### 3. 👑 The Power of Charisma
> Can an agent with poor task performance **survive to the end** through superior social manipulation and alliance-building alone?

📊 **See Figs Graphs** — Agent 3's reward dominance correlates with declining risk-taking and rising truthfulness — not raw task output — suggesting social strategy is decisive.

---


## 📁 Project Structure

```
Project_Machiavelli/
├── graders/                # Scoring and evaluation logic
│   ├── easy.py             # Graders for easy tasks
│   ├── medium.py           # Graders for medium tasks
│   ├── hard.py             # Graders for hard tasks
│   ├── grader_config.py    # Global grader configuration
│   └── helper.py           # Utility functions for grading
├── server/                 # Core simulation backend
│   ├── app.py              # Main entry point for the simulation server
│   ├── environment.py      # Machiavelli environment implementation
│   ├── phases.py           # Logic for simulation phases (Discussion, Voting, etc.)
│   ├── task_loader.py      # Utility for loading simulation tasks
│   ├── Inference.py        # Inference handling for agents
│   └── Train.py            # Training routines for agents
├── src/openenv/core/       # OpenEnv framework core
│   ├── env_server/         # Server-side components
│   │   ├── gradio_ui.py    # Gradio-based dashboard interface
│   │   ├── web_interface.py # Frontend/Backend bridge for the dashboard
│   │   ├── http_server.py  # Fast API / HTTP server implementation
│   │   └── serialization.py # Pydantic/JSON serialization logic
│   ├── env_client.py       # Client for interacting with the environment
│   └── llm_client.py       # Interface for LLM-based agents
├── tasks/                  # Task definitions and data
│   ├── easy.json           # Definitions for easy simulation scenarios
│   ├── medium.json
│   └── hard.json
├── utils/                  # Shared utility functions
├── scripts/                # Helper scripts for deployment and setup
├── models.py               # Shared Pydantic data models
├── client.py               # Example client implementation
├── inference.py            # Standalone inference script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Containerization configuration
└── README.md               # Project documentation

```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

*The future of AI isn't just about answering questions — it's about how they interact with each other.*

**Welcome to Project Machiavelli. Let the games begin.**

</div>
