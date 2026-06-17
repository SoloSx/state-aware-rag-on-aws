"""State-Aware RAG の高レベル API。

config に応じて Socratic / MCTS のいずれかで質問に回答する。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import Config, load_config
from .extractor import Extractor
from .generator import Generator
from .judge import Judge
from .llm import BedrockLLM
from .mcts import MCTSPlanner
from .retriever import build_retriever
from .socratic import SocraticPlanner

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    question: str
    answer: str
    mode: str
    details: dict


class StateAwareRAG:
    """論文 State-Aware RAG の推論パイプライン（AWS Bedrock 版）。"""

    def __init__(self, config: Config | None = None):
        self.cfg = config or load_config()
        self.llm = BedrockLLM(self.cfg.bedrock)
        self.generator = Generator(self.llm)
        self.extractor = Extractor(self.llm)
        self.judge = Judge(self.llm)
        self.retriever = build_retriever(self.cfg.retriever)

    def answer(self, question: str, mode: str | None = None) -> Answer:
        mode = mode or self.cfg.inference.mode

        if mode == "socratic":
            planner = SocraticPlanner(
                self.generator, self.retriever, self.extractor, self.cfg.inference
            )
            result = planner.run(question)
            return Answer(
                question=question,
                answer=result.answer,
                mode=mode,
                details={
                    "num_steps": len(result.steps),
                    "steps": [
                        {"sub_question": s.sub_question, "response": s.response} for s in result.steps
                    ],
                    "memory": result.memory.render(),
                    "retrieved_contexts": result.retrieved_contexts,
                },
            )

        if mode == "mcts":
            planner = MCTSPlanner(
                self.generator, self.retriever, self.extractor, self.judge, self.cfg.inference
            )
            result = planner.run(question)
            return Answer(
                question=question,
                answer=result.answer,
                mode=mode,
                details={
                    "best_score": result.best_score,
                    "num_nodes": result.num_nodes,
                    "memory": result.shared_memory.render(),
                },
            )

        raise ValueError(f"未知の mode: {mode}（socratic | mcts）")

    def evaluate(self, question: str, predicted: str, gold: str) -> bool:
        """ベンチマーク用 Acc 指標（論文の LLM-as-a-Judge）。"""
        return self.judge.evaluate_answer(question, predicted, gold)
