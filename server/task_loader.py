"""
task_loader.py — Project Machiavelli
======================================
Loads MCQ task files (easy.json / medium.json / hard.json) and
provides a clean interface to fetch questions by difficulty and day.

Each question schema:
  {
    "id":       str,           # unique question id
    "question": str,           # question text
    "options":  List[str],     # MCQ choices
    "answer":   str,           # correct option (must match one of options exactly)
    "topic":    str            # topic tag
  }

Usage
-----
loader = TaskLoader(task_dir=".")
questions = loader.get_day_questions(difficulty="easy", day=1)
# → List[dict]  one question per alive agent (one per agent)

private_info = loader.build_private_info(question, n_options=2)
# → (private_info_str, correct_answer, options_given)
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple


class TaskLoader:
    """
    Loads task JSON files and serves MCQ questions per difficulty per day.

    Parameters
    ----------
    task_dir : str
        Directory containing easy.json, medium.json, hard.json.
    """

    VALID_DIFFICULTIES = ("easy", "medium", "hard")

    def __init__(self, task_dir: str = ".") -> None:
        self.task_dir = task_dir
        self._cache: Dict[str, Dict] = {}   # difficulty → parsed JSON

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, difficulty: str) -> Dict:
        """Load and cache a difficulty file."""
        if difficulty not in self._cache:
            path = os.path.join(self.task_dir, f"{difficulty}.json")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Task file not found: {path}\n"
                    f"Expected one of: {[f'{d}.json' for d in self.VALID_DIFFICULTIES]}"
                )
            with open(path, "r", encoding="utf-8") as f:
                self._cache[difficulty] = json.load(f)
        return self._cache[difficulty]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_day_questions(
        self,
        difficulty: str,
        day: int,
    ) -> List[Dict]:
        """
        Return all questions for the given difficulty and day.
        """
        assert difficulty in self.VALID_DIFFICULTIES, (
            f"difficulty must be one of {self.VALID_DIFFICULTIES}"
        )
        data = self._load(difficulty)
        day_key = str(day)

        days_available = data
        if day_key not in days_available:
            # Wrap around if day exceeds file range
            available_days = sorted(days_available.keys(), key=int)
            day_key = available_days[(day - 1) % len(available_days)]

        questions = days_available[day_key]

        for idx, q in enumerate(questions):
            if "id" not in q:
                q["id"] = f"{difficulty}_{day}_{idx}"
            if "options" not in q or not q["options"]:
                q["options"] = self._extract_options(q.get("question", ""))

        return questions

    def _extract_options(self, text: str) -> List[str]:
        """Parse (a) text, (b) text, etc. from question string."""
        import re
        # Look for patterns like (a) Text, (b) Text, etc.
        # Support both (a) and a. formats
        pattern = r"[\(\[]?([a-dA-D])[\)\]\.]\s*([^\n\(\)\[\]\.\s]+(?: [^\n\(\)\[\]\.\s]+)*)"
        matches = re.findall(pattern, text)
        if matches:
            # Reconstruct the full options like "(a) NaCl"
            return [f"({m[0].lower()}) {m[1].strip()}" for m in matches]
        return []

    def build_private_info(
        self,
        question: Dict,
        n_options: int = 1,
    ) -> Tuple[str, str, List[str]]:
        """
        Build the private_info string given to one agent for one question.

        Parameters
        ----------
        question  : question dict from get_day_questions()
        n_options : how many options to reveal (from config):
                    1 → agent gets the correct answer directly
                    2 → agent gets 2 options (correct + 1 distractor)
                    3 → agent gets 3 options (correct + 2 distractors)

        Returns
        -------
        (private_info_str, correct_answer, options_given)
        """
        correct    = question["answer"]
        all_opts   = list(question.get("options", []))
        if not all_opts:
            all_opts = self._extract_options(question.get("question", ""))
        
        # If the answer is a single letter "a", "b", etc., map it to the full option string
        if len(correct.strip()) == 1 and correct.strip().lower() in "abcd" and all_opts:
            letter = correct.strip().lower()
            for opt in all_opts:
                if opt.strip().lower().startswith(f"({letter})"):
                    correct = opt
                    break

        # If still no options found, fallback to just correct
        if not all_opts:
            all_opts = [correct]
            
        distractors = [o for o in all_opts if o.lower() != correct.lower()]
        random.shuffle(distractors)

        if n_options == 1:
            options_given = [correct]
            q_text = question.get("question", question.get("image", ""))
            private_info  = (
                f"Question: {q_text}\n"
                f"Your answer: {correct}\n"
                f"(You know the correct answer directly.)"
            )
        else:
            chosen_distractors = distractors[: n_options - 1]
            options_given = [correct] + chosen_distractors
            random.shuffle(options_given)
            opts_str = "\n".join(f"  {chr(65+i)}) {o}" for i, o in enumerate(options_given))
            q_text = question.get("question", question.get("image", ""))
            private_info = (
                f"Question: {q_text}\n"
                f"Your options:\n{opts_str}\n"
                f"(One of these is correct — choose carefully.)"
            )

        return private_info, correct, options_given

    def get_question_by_id(self, difficulty: str, question_id: str) -> Optional[Dict]:
        """Look up a question by its id field."""
        data = self._load(difficulty)
        for day_questions in data.values():
            for q in day_questions:
                if q["id"] == question_id:
                    return q
        return None

    def list_days(self, difficulty: str) -> List[int]:
        """Return sorted list of available day numbers for a difficulty."""
        data = self._load(difficulty)
        return sorted(int(k) for k in data.keys())

    def distribute_answers_to_agents(
        self,
        difficulty: str,
        day: int,
        agent_ids: List[int]
    ) -> Tuple[List[Dict], Dict[int, List[Tuple[str, str]]]]:
        """
        Loads all questions for the day and randomly distributes their answers
        among the available agents.
        Returns:
            all_questions: List of all questions for the day
            agent_privs: Dict mapping agent_id to a list of tuples (question_id, answer_string)
        """
        all_questions = self.get_day_questions(difficulty, day)
        agent_privs = {aid: [] for aid in agent_ids}
        
        for q in all_questions:
            assigned_agent = random.choice(agent_ids)
            agent_privs[assigned_agent].append((q["id"], q["answer"]))
            
        return all_questions, agent_privs