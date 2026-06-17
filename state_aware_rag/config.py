"""State-Aware RAG の設定。

論文 "Reasoning with Memory: Adaptive Information Management for RAG" を
AWS 上で再現するための一元的な設定。環境変数で上書きできる。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


@dataclass
class BedrockConfig:
    """Bedrock(LLM) 設定。

    論文では Generator=Qwen3-8B, Extractor=Qwen3-4B を使うが、
    Table 3 で「Claude をそのまま Extractor に使う」構成が最高性能。
    AWS 上で学習なしに再現するため Bedrock の Claude を既定にする。
    """

    region: str = field(default_factory=lambda: _env("AWS_REGION", "ap-northeast-1"))

    # 推論プロファイル: global.* を既定にする（geo比 約10%安い。データ所在要件が
    # あれば jp.anthropic.claude-sonnet-4-5-... に差し替え）。東京に 3.7 Sonnet は
    # 無いため Judge も Sonnet 4.5 に統一。
    # Generator: 中間回答・サブ質問生成（凍結コンポーネント）
    generator_model_id: str = field(
        default_factory=lambda: _env("SAR_GENERATOR_MODEL", "global.anthropic.claude-sonnet-4-5-20250929-v1:0")
    )
    # Extractor: working memory の filter / consolidate / update（本来は学習対象）
    extractor_model_id: str = field(
        default_factory=lambda: _env("SAR_EXTRACTOR_MODEL", "global.anthropic.claude-sonnet-4-5-20250929-v1:0")
    )
    # Judge: MCTS の報酬計算 / 評価
    judge_model_id: str = field(
        default_factory=lambda: _env("SAR_JUDGE_MODEL", "global.anthropic.claude-sonnet-4-5-20250929-v1:0")
    )

    # 論文 Appendix B.3: generator は temperature=1, それ以外は 0.1, top_p=0.9, max 4096 tokens
    generator_temperature: float = field(default_factory=lambda: _env_float("SAR_GEN_TEMP", 1.0))
    other_temperature: float = field(default_factory=lambda: _env_float("SAR_OTHER_TEMP", 0.1))
    top_p: float = field(default_factory=lambda: _env_float("SAR_TOP_P", 0.9))
    max_tokens: int = field(default_factory=lambda: _env_int("SAR_MAX_TOKENS", 4096))


@dataclass
class RetrieverConfig:
    """Retriever 設定。

    既定は Amazon Bedrock Knowledge Base (OpenSearch Serverless + Titan Embeddings)。
    ローカル検証用に FAISS バックエンドも選べる。
    """

    backend: str = field(default_factory=lambda: _env("SAR_RETRIEVER_BACKEND", "bedrock_kb"))  # bedrock_kb | faiss
    knowledge_base_id: str = field(default_factory=lambda: _env("SAR_KB_ID", ""))
    region: str = field(default_factory=lambda: _env("AWS_REGION", "ap-northeast-1"))

    # 論文 Table 7: Retriever Top-k = 5
    top_k: int = field(default_factory=lambda: _env_int("SAR_RETRIEVER_TOPK", 5))

    # FAISS バックエンド用
    faiss_index_path: str = field(default_factory=lambda: _env("SAR_FAISS_INDEX", "./data/faiss.index"))
    faiss_docs_path: str = field(default_factory=lambda: _env("SAR_FAISS_DOCS", "./data/docs.jsonl"))
    embedding_model_id: str = field(
        default_factory=lambda: _env("SAR_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
    )


@dataclass
class InferenceConfig:
    """推論ハイパーパラメータ（論文 Table 7 / Section 5.3）。"""

    mode: str = field(default_factory=lambda: _env("SAR_MODE", "socratic"))  # socratic | mcts

    # アブレーション（論文 Table 2 相当）。反復メモリの貢献を切り分けるための設定。
    #   full             : Consolidation + Memory Update（通常）
    #   no_memory_update : 各ステップで consolidate するが memory を持ち越さない（反復メモリOFF）
    #   no_consolidation : Extractor の絞り込みをせず生文書を渡す（memory は持ち越す）
    #   no_extractor     : 絞り込みも持ち越しもしない（生文書蓄積＝IR-CoT 相当）
    ablation: str = field(default_factory=lambda: _env("SAR_ABLATION", "full"))

    # 論文 Appendix B.3 / Section 5.3: 飽和点が 5 step
    max_reasoning_steps: int = field(default_factory=lambda: _env_int("SAR_MAX_STEPS", 5))

    # MCTS 専用 (Table 7)
    max_depth: int = field(default_factory=lambda: _env_int("SAR_MCTS_DEPTH", 5))
    num_rollouts: int = field(default_factory=lambda: _env_int("SAR_MCTS_ROLLOUTS", 10))
    expansion_top_k: int = field(default_factory=lambda: _env_int("SAR_MCTS_EXPAND_K", 3))
    exploration_weight: float = field(default_factory=lambda: _env_float("SAR_MCTS_C", 1.0))


@dataclass
class Config:
    bedrock: BedrockConfig = field(default_factory=BedrockConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def load_config() -> Config:
    return Config()
