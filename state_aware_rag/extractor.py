"""Extractor（論文で唯一学習されるコンポーネント、ここでは prompt-only 実装）。

論文 Eq.(1)(2) の 2 フェーズを実装する:
  (1) Consolidation: I_i = E_consolidate(q_i, D_i, M_{i-1})
      retrieved documents から関連情報を抽出し、矛盾解消・重複排除を行う。
  (2) Memory Update: M_i = E_update(x, M_{i-1}, I_i, (q_i, r_i))
      新しい QA ペアと蒸留知識を memory に統合する。

論文 Table 3 では「Claude をそのまま Extractor に使う」構成が最高性能のため、
AWS 上では Bedrock の Claude を学習なしで Extractor として用いる（prompt-only）。
SFT/GRPO で学習した Qwen3-4B を SageMaker にデプロイした場合は、
config の extractor_model_id をそのエンドポイント ID に差し替えるだけで置換可能。
"""
from __future__ import annotations

import logging

from . import prompts
from .llm import BedrockLLM
from .memory import QAPair, WorkingMemory
from .retriever import Document

logger = logging.getLogger(__name__)


class Extractor:
    def __init__(self, llm: BedrockLLM):
        self.llm = llm

    # -- Eq.(1) Consolidation ----------------------------------------------
    def consolidate(self, sub_question: str, documents: list[Document], memory: WorkingMemory) -> str:
        """retrieved documents + 現 memory から関連情報 I_i を蒸留する。

        Extract prompt を各文書に適用し、relevant と判定された抽出を結合する。
        """
        relevant_chunks: list[str] = []

        # 現在の memory も「文書」として再評価対象に含める（reflecting on memory）
        sources = list(documents)
        if not memory.is_empty():
            sources.insert(0, Document(text=memory.render(), source="working_memory"))

        for doc in sources:
            if not doc.text.strip():
                continue
            raw = self.llm.extract(
                prompts.EXTRACT_PROMPT.format(question=sub_question, document=doc.text)
            )
            parsed = self.llm.parse_json(raw)
            if parsed.get("decision") == "relevant":
                info = parsed.get("extracted_information", "").strip()
                if info:
                    relevant_chunks.append(info)

        distilled = "\n".join(relevant_chunks)
        logger.debug("consolidate -> %d relevant chunks", len(relevant_chunks))
        return distilled

    # -- Eq.(2) Memory Update ----------------------------------------------
    def update(
        self,
        original_question: str,
        memory: WorkingMemory,
        distilled_info: str,
        qa_pair: QAPair,
    ) -> WorkingMemory:
        """(q_i, r_i) と I_i を M_{i-1} に統合して M_i を作る。

        Extract prompt を「memory 全体 + 新規情報」に対して適用し、
        original question 観点で filter & consolidate した結果を新 memory とする。
        """
        combined = "\n\n".join(
            [
                f"### Existing Working Memory\n{memory.important_information}" if memory.important_information else "",
                f"### Newly Distilled Information\n{distilled_info}" if distilled_info else "",
                f"### New Q/A\nSub-Q: {qa_pair.sub_question}\nSub-A: {qa_pair.response}",
            ]
        ).strip()

        raw = self.llm.extract(
            prompts.EXTRACT_PROMPT.format(question=original_question, document=combined)
        )
        parsed = self.llm.parse_json(raw)
        consolidated = parsed.get("extracted_information", "").strip()

        new_memory = memory.clone()
        # consolidate に失敗した場合は情報損失を避け、素朴に append する
        new_memory.important_information = consolidated or (
            (memory.important_information + "\n" + distilled_info).strip()
        )
        new_memory.qa_pairs = memory.qa_pairs + [qa_pair]
        return new_memory
