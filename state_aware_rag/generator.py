"""Generator（凍結コンポーネント）。

論文の primitive operations のうち、Generator が担う処理をまとめる:
  - サブ質問生成 (Sub-question prompt, A1)
  - クエリ生成 (Query Generation prompt)
  - 中間回答生成 (Answer prompt, A1)
  - Consolidate (A2) / Refine (A3) / Redirect (A4) / Finalize (A5)
"""
from __future__ import annotations

from dataclasses import dataclass

from . import prompts
from .llm import BedrockLLM


@dataclass
class SubQuestionResult:
    answerable: bool
    subquestion: str


class Generator:
    def __init__(self, llm: BedrockLLM):
        self.llm = llm

    # -- A1: サブ質問生成 ---------------------------------------------------
    def generate_subquestion(self, main_question: str, context: str) -> SubQuestionResult:
        raw = self.llm.generate(
            prompts.SUBQUESTION_PROMPT.format(question=main_question, context=context or "(none)")
        )
        p = self.llm.parse_json(raw)
        return SubQuestionResult(
            answerable=bool(p.get("answerable", False)),
            subquestion=p.get("subquestion", "").strip(),
        )

    # -- 検索クエリ生成 -----------------------------------------------------
    def generate_queries(self, question: str) -> list[str]:
        raw = self.llm.generate(prompts.QUERY_GENERATION_PROMPT.format(question=question))
        p = self.llm.parse_json(raw)
        queries = p.get("queries", [])
        return [q for q in queries if isinstance(q, str) and q.strip()] or [question]

    # -- A1: 中間回答生成 ---------------------------------------------------
    def answer(self, question: str, context: str) -> str:
        raw = self.llm.generate(
            prompts.ANSWER_PROMPT.format(question=question, context=context or "(none)")
        )
        p = self.llm.parse_json(raw)
        return p.get("answer", "").strip() or raw.strip()

    # -- A2: Consolidate ----------------------------------------------------
    def consolidate(self, question: str, context: str) -> str:
        raw = self.llm.generate(
            prompts.CONSOLIDATE_PROMPT.format(question=question, context=context or "(none)")
        )
        p = self.llm.parse_json(raw)
        return p.get("answer", "").strip() or raw.strip()

    # -- A3: Refine ---------------------------------------------------------
    def refine(self, question: str, answer: str, context: str) -> str:
        raw = self.llm.generate(
            prompts.REFINE_PROMPT.format(question=question, answer=answer, context=context or "(none)")
        )
        p = self.llm.parse_json(raw)
        return p.get("answer", "").strip() or answer

    # -- A4: Redirect -------------------------------------------------------
    def redirect(self, question: str) -> str:
        raw = self.llm.generate(prompts.REDIRECT_PROMPT.format(question=question))
        p = self.llm.parse_json(raw)
        return p.get("rephrased_question", "").strip() or question

    # -- A5: Finalize -------------------------------------------------------
    def finalize(self, question: str, context: str) -> str:
        raw = self.llm.generate(
            prompts.FINALIZE_PROMPT.format(question=question, context=context or "(none)")
        )
        p = self.llm.parse_json(raw)
        return p.get("answer", "").strip() or raw.strip()
