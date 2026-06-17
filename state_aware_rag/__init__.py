"""State-Aware RAG — AWS Bedrock 版の再現実装。

論文: "Reasoning with Memory: Adaptive Information Management for
Retrieval-Augmented Generation" (Man et al., AWS AI)
"""
from .config import Config, load_config
from .pipeline import Answer, StateAwareRAG

__all__ = ["Config", "load_config", "StateAwareRAG", "Answer"]
__version__ = "0.1.0"
