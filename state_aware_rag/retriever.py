"""Retriever（凍結コンポーネント）。

論文では Qwen3-Embedding-4B + FAISS で Wikipedia 2023 dump を検索する。
AWS 上での再現では 2 つのバックエンドを提供する:
  1. bedrock_kb : Amazon Bedrock Knowledge Base ── マネージドで本番向け。
                  ベクトルストアは S3 Vectors / OpenSearch Serverless 等いずれでも
                  Retrieve API は同一（このコードはバックエンド非依存）。
  2. faiss      : ローカル FAISS + Titan Embeddings ── 小規模検証・オフライン用。
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import boto3

from .config import RetrieverConfig

logger = logging.getLogger(__name__)


@dataclass
class Document:
    text: str
    score: float = 0.0
    source: str = ""


class BaseRetriever:
    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        raise NotImplementedError


class BedrockKBRetriever(BaseRetriever):
    """Amazon Bedrock Knowledge Base Retrieve API を使う。"""

    def __init__(self, cfg: RetrieverConfig):
        if not cfg.knowledge_base_id:
            raise ValueError("SAR_KB_ID (knowledge_base_id) が未設定です。")
        self.cfg = cfg
        self.client = boto3.client("bedrock-agent-runtime", region_name=cfg.region)

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        k = top_k or self.cfg.top_k
        resp = self.client.retrieve(
            knowledgeBaseId=self.cfg.knowledge_base_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": k}},
        )
        docs: list[Document] = []
        for r in resp.get("retrievalResults", []):
            docs.append(
                Document(
                    text=r.get("content", {}).get("text", ""),
                    score=r.get("score", 0.0),
                    source=r.get("location", {}).get("s3Location", {}).get("uri", ""),
                )
            )
        return docs


class FaissRetriever(BaseRetriever):
    """ローカル FAISS インデックス + Bedrock Titan Embeddings。

    ingest/build_faiss_index.py で作成したインデックスを読み込む。
    """

    def __init__(self, cfg: RetrieverConfig):
        import faiss  # 遅延 import（bedrock_kb 利用時は不要）

        self.cfg = cfg
        self.bedrock = boto3.client("bedrock-runtime", region_name=cfg.region)
        if not os.path.exists(cfg.faiss_index_path):
            raise FileNotFoundError(
                f"FAISS index not found: {cfg.faiss_index_path}. "
                "ingest/build_faiss_index.py を先に実行してください。"
            )
        self.index = faiss.read_index(cfg.faiss_index_path)
        with open(cfg.faiss_docs_path, encoding="utf-8") as f:
            self.docs = [json.loads(line) for line in f]

    def _embed(self, text: str) -> list[float]:
        resp = self.bedrock.invoke_model(
            modelId=self.cfg.embedding_model_id,
            body=json.dumps({"inputText": text}),
        )
        return json.loads(resp["body"].read())["embedding"]

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        import numpy as np

        k = top_k or self.cfg.top_k
        vec = np.array([self._embed(query)], dtype="float32")
        faiss_norm(vec)
        scores, idxs = self.index.search(vec, k)
        out: list[Document] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            d = self.docs[idx]
            out.append(Document(text=d["text"], score=float(score), source=d.get("id", "")))
        return out


def faiss_norm(vec) -> None:
    """L2 正規化（cosine 類似度を内積で計算するため）。"""
    import numpy as np

    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vec /= norms


class WebSearchRetriever(BaseRetriever):
    """Web 検索リトリーバ（鍵不要の DuckDuckGo を使用）。

    論文 Section 5.2 の Component Analysis では、静的 Wikipedia コーパスを
    Google Search に置き換えると Bamboogle で +22.2% という結果が出ている。
    そのオープン検索の代替として DuckDuckGo を使い、Bamboogle のような
    一般知識マルチホップ質問でも実コーパス相当の検索を可能にする。
    """

    def __init__(self, cfg: RetrieverConfig):
        self.cfg = cfg

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        from ddgs import DDGS

        k = top_k or self.cfg.top_k
        docs: list[Document] = []
        try:
            with DDGS() as ddgs:
                for i, r in enumerate(ddgs.text(query, max_results=k)):
                    text = f"{r.get('title', '')}. {r.get('body', '')}".strip()
                    if text:
                        docs.append(Document(text=text, score=1.0 - i * 0.05, source=r.get("href", "")))
        except Exception as e:  # 検索失敗時は空（Generator が自力で補完）
            logger.warning("web search failed for %r: %s", query, e)
        return docs


def build_retriever(cfg: RetrieverConfig) -> BaseRetriever:
    if cfg.backend == "bedrock_kb":
        return BedrockKBRetriever(cfg)
    if cfg.backend == "faiss":
        return FaissRetriever(cfg)
    if cfg.backend == "web":
        return WebSearchRetriever(cfg)
    raise ValueError(f"未知の retriever backend: {cfg.backend}")
