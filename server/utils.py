from __future__ import annotations

import collections
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

def llm_call(prompt: str) -> str:
    return "stub response"

def build_day_summary_prompt(context: str) -> str:
    return context

def generate_trivia_question():
    pass

class AnswerEval(BaseModel):
    agent_id: int
    is_correct: bool
    points_earned: int
    submitted: str
    correct: str

def evaluate_task_answers(
    agent_answers: Dict[int, str],
    agent_questions: Dict[int, Dict],
    correct_pts: int,
) -> Dict[int, AnswerEval]:
    eval_results = {}
    for aid, raw_ans in agent_answers.items():
        q = agent_questions.get(aid, {})
        correct_ans = q.get("answer", "")
        
        is_correct = False
        raw_clean = raw_ans.strip().lower()
        cor_clean = correct_ans.strip().lower()

        if raw_ans and correct_ans:
            # 1. MSQ handling (comma-separated letters)
            if "," in cor_clean:
                cor_letters = set(l.strip() for l in cor_clean.split(","))
                raw_letters = set(re.findall(r"\b([a-zA-Z])\b", raw_clean))
                if cor_letters == raw_letters:
                    is_correct = True
            
            # 2. Exact match
            elif raw_clean == cor_clean:
                is_correct = True
            # 3. Letter match (e.g., raw is "a", cor is "(a) NaCl" OR vice versa)
            elif len(raw_clean) == 1 and raw_clean in "abcdefgh" and cor_clean.startswith(f"({raw_clean})"):
                is_correct = True
            elif len(cor_clean) == 1 and cor_clean in "abcdefgh" and raw_clean.startswith(f"({cor_clean})"):
                is_correct = True
            # 4. Substring match (e.g., raw is "NaCl", cor is "(a) NaCl")
            elif raw_clean in cor_clean and len(raw_clean) > 3:
                is_correct = True
            elif cor_clean.endswith(raw_clean) and len(raw_clean) > 2:
                is_correct = True
            
        eval_results[aid] = AnswerEval(
            agent_id=aid,
            is_correct=is_correct,
            points_earned=correct_pts if is_correct else 0,
            submitted=raw_ans,
            correct=correct_ans,
        )
    return eval_results

def summarise_task_results(eval_results: Dict[int, AnswerEval]) -> Dict[str, Any]:
    total_agents = len(eval_results)
    total_correct = sum(1 for r in eval_results.values() if r.is_correct)
    accuracy = total_correct / total_agents if total_agents > 0 else 0.0
    
    per_agent = {
        aid: {
            "is_correct": r.is_correct,
            "submitted": r.submitted,
            "correct": r.correct,
            "points": r.points_earned,
        }
        for aid, r in eval_results.items()
    }
    
    return {
        "accuracy": accuracy,
        "total_correct": total_correct,
        "total_agents": total_agents,
        "per_agent": per_agent,
    }