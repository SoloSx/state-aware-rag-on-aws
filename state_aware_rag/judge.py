"""LLM-as-a-Judge（論文 Eq.(3)(4) と評価指標）。

- path_reward    : 局所的な推論品質 R_p (Eq.3) ── RL 学習・分析用
- outcome_reward : 全体の推論成功 R_o (Eq.4)
- judge_answer   : MCTS のノード報酬 (0-10)
- evaluate_answer: ベンチマークの Acc 指標（二値正誤、論文では Claude 3.7 Sonnet）
"""
from __future__ import annotations

from . import prompts
from .llm import BedrockLLM


class Judge:
    def __init__(self, llm: BedrockLLM):
        self.llm = llm

    def _score(self, raw: str) -> float:
        p = self.llm.parse_json(raw)
        try:
            return float(p.get("score", 0.0))
        except (TypeError, ValueError):
            return 0.0

    # -- Eq.(3) Path Reward -------------------------------------------------
    def path_reward(
        self,
        main_question: str,
        reasoning_trace: str,
        sub_question: str,
        selected_information: str,
        generated_answer: str,
    ) -> float:
        raw = self.llm.judge(
            prompts.PATH_REWARD_PROMPT.format(
                main_question=main_question,
                reasoning_trace=reasoning_trace or "(none)",
                sub_question=sub_question,
                selected_information=selected_information or "(none)",
                generated_answer=generated_answer,
            )
        )
        return self._score(raw)

    # -- Eq.(4) Outcome Reward ---------------------------------------------
    def outcome_reward(self, original_question: str, reasoning_path: str, correct_answer: str = "") -> float:
        raw = self.llm.judge(
            prompts.OUTCOME_REWARD_PROMPT.format(
                original_question=original_question,
                reasoning_path=reasoning_path,
                correct_answer=correct_answer or "(not provided)",
            )
        )
        return self._score(raw)

    # -- MCTS ノード報酬 ----------------------------------------------------
    def judge_answer(self, user_question: str, system_answer: str, correct_answer: str = "") -> float:
        raw = self.llm.judge(
            prompts.JUDGE_ANSWER_PROMPT.format(
                user_question=user_question,
                system_answer=system_answer,
                correct_answer=correct_answer or "(not provided)",
            )
        )
        return self._score(raw)

    # -- ベンチマーク Acc 指標 ---------------------------------------------
    def evaluate_answer(self, question: str, predicted_answer: str, correct_answer: str) -> bool:
        raw = self.llm.judge(
            prompts.EVALUATE_ANSWER_PROMPT.format(
                question=question,
                correct_answer=correct_answer,
                predicted_answer=predicted_answer,
            )
        )
        return bool(self.llm.parse_json(raw).get("correct", False))

    # -- 複数 path の統合 ---------------------------------------------------
    def synthesize(self, question: str, reasoning_paths: str) -> str:
        raw = self.llm.judge(
            prompts.SYNTHESIZE_PROMPT.format(question=question, reasoning_paths=reasoning_paths)
        )
        return self.llm.parse_json(raw).get("answer", "").strip() or raw.strip()
