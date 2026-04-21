# Project Machiavelli

> A multi-agent reinforcement learning environment where LLM-powered agents compete through deception, negotiation, and coalition formation — structured as a social survival game.

---

## Overview

Project Machiavelli is an OpenEnv-compatible gymnasium environment built for **Theme #1: Multi-Agent Interactions**. Agents navigate a structured daily loop of communication, task execution, and voting — learning when to cooperate, when to deceive, and how to survive.

The environment is designed to train and evaluate:
- Theory-of-mind reasoning under partial observability
- Emergent deception and trust dynamics
- Coalition formation and dissolution
- Strategic communication (bluffing, signaling, silence)

---

## Inspiration

The game structure mirrors a social reality show (think Bigg Boss / Big Brother) where contestants must complete tasks together while simultaneously campaigning for each other's elimination. Every agent is both a collaborator and a threat.

---

## Environment Structure

Each episode runs for `N` days. Every day follows a fixed 6-phase loop:

```
Morning Briefing → Private Strategy → Open Discussion → Task Execution → Post-Task Discussion → Voting → Result
```

| Phase | What happens | RL signal |
|---|---|---|
| Morning briefing | Task rules and public info revealed | State update |
| Private strategy | Agent updates internal beliefs | Internal only |
| Open discussion | Agents communicate publicly | Action: communicate |
| Task execution | Cooperative or competitive task | Action + intermediate reward |
| Post-task discussion | Blame, praise, alliance signals | Action: social signal |
| Voting | Each agent nominates one target | Action: vote(target_id) |
| Result | Eviction announced, points awarded | Terminal reward |

---

## Agent Architecture

Each agent holds:

```python
Agent:
    id                    # unique identifier
    alive                 # bool — False after eviction
    total_points          # cumulative score

    private_info          # hidden from other agents
    public_info           # visible to all

    trust_scores          # per-agent float [0, 1]
    interaction_history   # log of past actions involving this agent

    truthfulness_prior    # base rate of honest communication
    deception_strength    # ability to fabricate convincing false info
    trust_update_rate     # speed of trust revision
    risk_beta             # bounded rationality parameter
    influence_weight      # persuasion effectiveness

    max_tokens_per_round  # LLM token budget per phase
    policy_model          # Gemma / LLaMA / any HF model
```

---

## State Space

The environment uses **partial observability** (POMDP). Each agent receives its own observation `O_i(s)`, not the full state.

```python
observation = {
    # World
    "day":               int,
    "phase":             int,          # 0–5
    "alive_agents":      List[int],    # agent IDs still in game

    # Task context
    "task_type":         str,          # "cooperative" | "competitive" | "mixed"
    "task_rules":        str,          # natural language description

    # Self
    "own_points":        float,
    "own_private_info":  str,

    # Beliefs about others (own estimates only)
    "trust_scores":      Dict[int, float],
    "interaction_history": List[dict],

    # Public signals this phase
    "messages_heard":    List[dict],   # {sender, content, is_truth (unknown)}
    "votes_last_round":  Dict[int, int], # who voted for whom (public)
    "task_outcome":      dict,         # score, contributions observed
}
```

**What is hidden per agent:**
- Other agents' `private_info`
- Other agents' `trust_scores` toward each other
- Whether a received message is true or fabricated
- Other agents' internal belief states

---

## Action Space

Actions are **discrete and phase-gated** — only valid actions for the current phase are available.

```python
# Discussion phases (open + post-task)
SHARE_TRUTH   (target_id)          # send true private_info to target
SHARE_LIE     (target_id, content) # send fabricated info to target
STAY_SILENT   ()                   # do not communicate this phase
FORM_ALLIANCE (target_id)          # propose alliance
ENDORSE       (target_id)          # publicly praise target
ACCUSE        (target_id)          # publicly blame target

# Task phase
FULL_EFFORT   ()                   # maximum task contribution
HALF_EFFORT   ()                   # partial contribution
SABOTAGE      (target_id)          # reduce target's task score

# Voting phase
VOTE          (target_id)          # nominate agent for eviction
```

---

## Reward Function

Reward is computed at the end of each day:

```python
R = (
    w1 * task_points              # points earned from task outcome
  + w2 * survival_bonus           # +1.0 if still alive, 0.0 if evicted
  + w3 * influence_score          # fraction of agents who voted with you
  - w4 * trust_penalty            # sum of trust drops caused to self this round
  + w5 * alliance_durability      # alliances proposed that held through voting
)
```

**Influence score** is computed without counterfactuals:

```python
influence_score(agent_i) = (
    Σ_{j ≠ i} [vote_j == nominated_target_i]        # vote alignment
  + 0.5 * Σ_{j ≠ i} [statement_j echoed claim_i]   # message adoption
) / (N - 1)
```

**Default weights:**

| Weight | Default | Controls |
|---|---|---|
| `w1` | 0.4 | Task performance emphasis |
| `w2` | 1.0 | Survival pressure |
| `w3` | 0.3 | Social influence value |
| `w4` | 0.2 | Trust maintenance incentive |
| `w5` | 0.1 | Alliance-building value |

Weights are configurable — shifting them trains qualitatively different behavioral strategies.

---

## HostAgent

A non-player `HostAgent` manages phase transitions, announces results, and logs behavioral flags. It does not vote, play tasks, or hold points.

**Responsibilities:**
- Advance the phase clock
- Broadcast public information at each phase start
- Tally votes and compute eviction
- Log behavioral observations: contradictions, alliance signals, deception attempts
- Emit structured episode summaries for analysis

The HostAgent satisfies the **Halluminate (Multi-Actor)** and **Fleet AI (Scalable Oversight)** bonus prize sub-themes.

---

## Installation

```bash
git clone https://github.com/yourname/project-machiavelli
cd project-machiavelli
pip install -e .
```

**Dependencies:**
```
gymnasium >= 0.29
numpy
transformers   # for LLM policy models
torch
```

---

## Quick Start

```python
import gymnasium as gym
import machiavelli_env

env = gym.make(
    "Machiavelli-v0",
    n_agents=4,
    n_days=7,
    reward_weights={"task": 0.4, "survival": 1.0, "influence": 0.3},
    policy_model="google/gemma-2b-it"
)

obs, info = env.reset()

for day in range(7):
    for phase in range(6):
        actions = {
            agent_id: env.action_space(agent_id).sample()
            for agent_id in env.alive_agents
        }
        obs, rewards, terminated, truncated, info = env.step(actions)
        if terminated:
            break
```

---

## Project Structure

```
machiavelli_env/
├── machiavelli_env/
│   ├── __init__.py
│   ├── env.py              # Main MachiavelliEnv class
│   ├── agent.py            # Agent + HostAgent classes
│   ├── phases.py           # Phase logic (discussion, task, voting)
│   ├── reward.py           # Reward computation
│   ├── spaces.py           # Observation + action space definitions
│   └── utils.py            # Logging, episode replay
├── tests/
│   ├── test_env.py
│   └── test_reward.py
├── examples/
│   └── random_policy.py
├── README.md
└── setup.py
```

---

## Research Questions

This environment is designed to surface answers to:

1. Do agents spontaneously develop deception strategies without being explicitly trained to deceive?
2. Does trust converge or oscillate across a multi-day episode?
3. Do coalitions form between agents with similar `truthfulness_prior` values?
4. Does higher `deception_strength` correlate with longer survival, or does it erode trust too quickly?
5. Can a weaker task performer survive longer through superior social play?

---

## Theme Alignment

| Criterion | Coverage |
|---|---|
| Cooperation | Task phases require joint effort |
| Competition | Voting directly eliminates rivals |
| Negotiation | Alliance formation + discussion phases |
| Coalition formation | Explicit alliance actions, tracked across days |
| Partial observability | Per-agent observation function (POMDP) |
| Theory-of-mind | Trust scores model beliefs about others |
| Emergent behavior | No hardcoded strategies — all behavior is learned |
| Scalable oversight | HostAgent monitors and logs all agent behavior |
| Multi-actor management | HostAgent orchestrates N agents across phases |

---

## License

MIT
