"""Socratic Planning（論文 Section 3.3 / 効率重視モード）。

action space を A1 (Decompose & Answer) と A5 (Conclude) に限定し、
単一の推論チェーンを生成する。IR-CoT 的に質問を 1 つずつ sub-question に
分解しつつ、各ステップで Extractor が memory を filter/consolidate するため、
長いチェーンでも context 汚染を防げる。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import InferenceConfig
from .extractor import Extractor
from .generator import Generator
from .memory import QAPair, WorkingMemory
from .retriever import BaseRetriever

logger = logging.getLogger(__name__)


@dataclass
class StepTrace:
    sub_question: str
    distilled_info: str
    response: str


@dataclass
class SocraticResult:
    answer: str
    steps: list[StepTrace] = field(default_factory=list)
    memory: WorkingMemory = field(default_factory=WorkingMemory)
    # RAGAS の context_precision / recall 用に、全ステップで検索した生の文書を保持
    retrieved_contexts: list[str] = field(default_factory=list)


class SocraticPlanner:
    def __init__(
        self,
        generator: Generator,
        retriever: BaseRetriever,
        extractor: Extractor,
        cfg: InferenceConfig,
    ):
        self.gen = generator
        self.ret = retriever
        self.ext = extractor
        self.cfg = cfg

    def run(self, question: str) -> SocraticResult:
        memory = WorkingMemory()
        steps: list[StepTrace] = []
        all_contexts: list[str] = []

        for step in range(self.cfg.max_reasoning_steps):
            # --- 終了判定: memory で答えられるか? (A1 の Decision Point) ---
            sq = self.gen.generate_subquestion(question, memory.render())
            if sq.answerable or not sq.subquestion:
                logger.info("step %d: answerable -> conclude", step)
                break

            sub_q = sq.subquestion
            logger.info("step %d: sub-question = %s", step, sub_q)

            # --- A1: Retrieve --- 検索クエリを生成して文書取得 ---
            queries = self.gen.generate_queries(sub_q)
            docs = []
            seen = set()
            for q in queries:
                for d in self.ret.retrieve(q):
                    if d.text not in seen:
                        seen.add(d.text)
                        docs.append(d)
                        if d.text not in all_contexts:
                            all_contexts.append(d.text)

            ablation = self.cfg.ablation

            # --- A1: Consolidate (Eq.1) --- 関連情報 I_i を蒸留 ---
            # アブレーション: no_consolidation / no_extractor は絞り込みを行わず生文書を渡す
            if ablation in ("no_consolidation", "no_extractor"):
                distilled = "\n\n".join(d.text for d in docs)
            else:
                distilled = self.ext.consolidate(sub_q, docs, memory)

            # --- A1: Answer --- 蒸留情報 + memory で中間回答 r_i ---
            context = "\n\n".join(filter(None, [memory.render(), distilled]))
            response = self.gen.answer(sub_q, context)

            # --- A1: Memory Update (Eq.2) --- M_i を更新 ---
            # アブレーション: no_memory_update / no_extractor は memory を持ち越さない（反復メモリOFF）
            if ablation in ("no_memory_update", "no_extractor"):
                pass  # memory は空のまま据え置き → Conclude 時に過去の発見を参照できない
            else:
                memory = self.ext.update(question, memory, distilled, QAPair(sub_q, response))
            steps.append(StepTrace(sub_q, distilled, response))

        # --- A5: Conclude --- 最終回答 ---
        final = self.gen.finalize(question, memory.render())
        return SocraticResult(answer=final, steps=steps, memory=memory, retrieved_contexts=all_contexts)
