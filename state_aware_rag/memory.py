"""Working Memory（論文の中核貢献）。

State-Aware RAG の memory M_i は「文書の受動的な蓄積」ではなく、推論に
関連する情報のみを保持する curated knowledge state。MCTS では木全体で
共有される globally-shared memory として振る舞う（branch 間で知見が伝播）。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class QAPair:
    """1 ステップの (sub-question, response) ペア。"""

    sub_question: str
    response: str


@dataclass
class WorkingMemory:
    """curated knowledge state M_i。

    - important_information: Extractor が consolidate/update した蒸留済み事実
    - qa_pairs: これまでの (q_i, r_i) の履歴
    """

    important_information: str = ""
    qa_pairs: list[QAPair] = field(default_factory=list)

    def render(self) -> str:
        """LLM の context として渡すための文字列表現。"""
        parts: list[str] = []
        if self.important_information.strip():
            parts.append(f"## Consolidated Working Memory\n{self.important_information.strip()}")
        if self.qa_pairs:
            history = "\n".join(
                f"- Sub-Q: {p.sub_question}\n  Sub-A: {p.response}" for p in self.qa_pairs
            )
            parts.append(f"## Prior Reasoning Steps\n{history}")
        return "\n\n".join(parts) if parts else "(empty)"

    def is_empty(self) -> bool:
        return not self.important_information.strip() and not self.qa_pairs

    def clone(self) -> "WorkingMemory":
        """MCTS のノード分岐用にディープコピー。"""
        return copy.deepcopy(self)
