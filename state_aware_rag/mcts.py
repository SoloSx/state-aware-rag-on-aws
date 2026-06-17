"""MCTS Exploration（論文 Section 3.3 / Figure 1 / Table 7）。

各ノードが推論状態 S_i を表し、エッジが 5 種類の action (A1-A5) に対応する。
UCT (Upper Confidence Bound for Trees) で木を辿り、Judge で報酬を計算する。

本実装の核心は globally-shared working memory: あるブランチで発見された
情報が共有 memory を更新し、他ブランチからも参照できる（cross-branch
knowledge transfer）。これが従来の MCTS-RAG（ブランチ独立）との違い。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from .config import InferenceConfig
from .extractor import Extractor
from .generator import Generator
from .judge import Judge
from .memory import QAPair, WorkingMemory
from .retriever import BaseRetriever

logger = logging.getLogger(__name__)

ACTIONS = ["A1", "A2", "A3", "A4", "A5"]


@dataclass
class MCTSNode:
    # 状態 S_i = (q_i, r_i, M_i)
    sub_question: str
    response: str
    memory: WorkingMemory
    action: str = "ROOT"  # このノードに至った action
    parent: "MCTSNode | None" = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0  # 累積報酬
    depth: int = 0
    terminal: bool = False
    final_answer: str = ""

    def uct(self, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.value / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits) if self.parent else 0.0
        return exploit + explore

    def best_child(self, c: float) -> "MCTSNode":
        return max(self.children, key=lambda n: n.uct(c))


@dataclass
class MCTSResult:
    answer: str
    best_score: float
    num_nodes: int
    shared_memory: WorkingMemory


class MCTSPlanner:
    def __init__(
        self,
        generator: Generator,
        retriever: BaseRetriever,
        extractor: Extractor,
        judge: Judge,
        cfg: InferenceConfig,
    ):
        self.gen = generator
        self.ret = retriever
        self.ext = extractor
        self.judge = judge
        self.cfg = cfg
        # globally-shared working memory（木全体で共有）
        self.shared_memory = WorkingMemory()
        self._node_count = 0

    def run(self, question: str) -> MCTSResult:
        self.question = question
        root = MCTSNode(sub_question=question, response="", memory=self.shared_memory.clone())
        self._node_count = 1

        for rollout in range(self.cfg.num_rollouts):
            logger.info("rollout %d/%d", rollout + 1, self.cfg.num_rollouts)
            node = self._select(root)
            if not node.terminal and node.depth < self.cfg.max_depth:
                node = self._expand(node)
            reward = self._simulate(node)
            self._backpropagate(node, reward)

        # 最良の終端回答を選ぶ
        best = self._best_terminal(root)
        if best is None:
            # 終端に到達しなかった場合は memory から finalize
            answer = self.gen.finalize(question, self.shared_memory.render())
            score = self.judge.judge_answer(question, answer)
        else:
            answer, score = best.final_answer or best.response, best.value / max(best.visits, 1)

        return MCTSResult(
            answer=answer,
            best_score=score,
            num_nodes=self._node_count,
            shared_memory=self.shared_memory,
        )

    # -- Selection ----------------------------------------------------------
    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children and not node.terminal:
            node = node.best_child(self.cfg.exploration_weight)
        return node

    # -- Expansion ----------------------------------------------------------
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """expansion_top_k 個の action でノードを展開し、最初の子を返す。"""
        actions = self._candidate_actions(node)[: self.cfg.expansion_top_k]
        for act in actions:
            child = self._apply_action(node, act)
            if child is not None:
                node.children.append(child)
                self._node_count += 1
        return node.children[0] if node.children else node

    def _candidate_actions(self, node: MCTSNode) -> list[str]:
        """状態に応じた action 候補。深さが上限に近ければ Conclude を優先。"""
        if node.depth >= self.cfg.max_depth - 1:
            return ["A5", "A3", "A2"]
        if node.response:
            # 既に回答がある -> Refine / Conclude / さらに分解 / Redirect
            return ["A1", "A3", "A5", "A2", "A4"]
        # まだ回答が無い -> まず分解して答える
        return ["A1", "A2", "A4", "A5"]

    def _apply_action(self, node: MCTSNode, action: str) -> MCTSNode | None:
        mem = node.memory.clone()
        depth = node.depth + 1

        if action == "A1":  # Decompose & Answer
            sq = self.gen.generate_subquestion(self.question, mem.render())
            if sq.answerable or not sq.subquestion:
                return self._conclude_node(node, mem, depth)
            sub_q = sq.subquestion
            docs = self._retrieve(sub_q)
            distilled = self.ext.consolidate(sub_q, docs, mem)
            context = "\n\n".join(filter(None, [mem.render(), distilled]))
            response = self.gen.answer(sub_q, context)
            mem = self.ext.update(self.question, mem, distilled, QAPair(sub_q, response))
            self._merge_shared(mem)  # globally-shared memory に反映
            return MCTSNode(sub_q, response, mem, action, node, depth=depth)

        if action == "A2":  # Consolidate（検索なし）
            synthesis = self.gen.consolidate(self.question, mem.render())
            mem.important_information = (mem.important_information + "\n" + synthesis).strip()
            self._merge_shared(mem)
            return MCTSNode(node.sub_question, synthesis, mem, action, node, depth=depth)

        if action == "A3":  # Refine
            refined = self.gen.refine(node.sub_question or self.question, node.response, mem.render())
            return MCTSNode(node.sub_question, refined, mem, action, node, depth=depth)

        if action == "A4":  # Redirect
            rephrased = self.gen.redirect(node.sub_question or self.question)
            return MCTSNode(rephrased, "", mem, action, node, depth=depth)

        if action == "A5":  # Conclude
            return self._conclude_node(node, mem, depth)

        return None

    def _conclude_node(self, node: MCTSNode, mem: WorkingMemory, depth: int) -> MCTSNode:
        answer = self.gen.finalize(self.question, mem.render())
        child = MCTSNode(node.sub_question, answer, mem, "A5", node, depth=depth)
        child.terminal = True
        child.final_answer = answer
        return child

    # -- Simulation (reward) ------------------------------------------------
    def _simulate(self, node: MCTSNode) -> float:
        """ノードの現時点の回答を Judge で 0-10 採点 → 0-1 に正規化。"""
        answer = node.final_answer or node.response
        if not answer:
            answer = self.gen.finalize(self.question, node.memory.render())
        score = self.judge.judge_answer(self.question, answer)
        return score / 10.0

    # -- Backpropagation ----------------------------------------------------
    def _backpropagate(self, node: MCTSNode | None, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    # -- helpers ------------------------------------------------------------
    def _retrieve(self, query: str):
        docs, seen = [], set()
        for q in self.gen.generate_queries(query):
            for d in self.ret.retrieve(q):
                if d.text not in seen:
                    seen.add(d.text)
                    docs.append(d)
        return docs

    def _merge_shared(self, mem: WorkingMemory) -> None:
        """ブランチで得た memory を globally-shared memory に統合する。"""
        if mem.important_information.strip():
            existing = self.shared_memory.important_information
            if mem.important_information not in existing:
                self.shared_memory.important_information = (
                    existing + "\n" + mem.important_information
                ).strip()
        # 既知でない QA ペアを追加
        known = {(p.sub_question, p.response) for p in self.shared_memory.qa_pairs}
        for p in mem.qa_pairs:
            if (p.sub_question, p.response) not in known:
                self.shared_memory.qa_pairs.append(p)

    def _best_terminal(self, root: MCTSNode) -> MCTSNode | None:
        best, best_score = None, -1.0
        stack = [root]
        while stack:
            n = stack.pop()
            stack.extend(n.children)
            if n.terminal and n.visits > 0:
                avg = n.value / n.visits
                if avg > best_score:
                    best, best_score = n, avg
        return best
