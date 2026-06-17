"""エンドツーエンドの動作例（論文 Figure 1 のマルチホップ質問）。

事前準備:
    1) Bedrock のモデルアクセスを有効化（Claude / Titan Embeddings）
    2) ローカル FAISS インデックスを構築:
         export SAR_RETRIEVER_BACKEND=faiss
         python ingest/build_faiss_index.py \
             --input examples/sample_corpus.jsonl \
             --index data/faiss.index --docs data/docs.jsonl
    3) 実行:
         python examples/run_example.py
"""
import logging

from state_aware_rag import StateAwareRAG

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# 論文 Figure 1 の例: 2 つの映画監督の出身国を比較するマルチホップ質問
QUESTION = (
    "Are the director of film Move (1970) and the director of film "
    "Mediterranee (1963) from the same country?"
)


def main() -> None:
    rag = StateAwareRAG()

    print("=" * 70)
    print("Socratic Planning（効率重視 / A1+A5）")
    print("=" * 70)
    result = rag.answer(QUESTION, mode="socratic")
    print(f"\nAnswer: {result.answer}")
    print(f"Steps : {result.details['num_steps']}")

    print("\n" + "=" * 70)
    print("MCTS Exploration（網羅探索 / A1-A5 + globally-shared memory）")
    print("=" * 70)
    result = rag.answer(QUESTION, mode="mcts")
    print(f"\nAnswer    : {result.answer}")
    print(f"Best score: {result.details['best_score']:.2f}")
    print(f"Nodes     : {result.details['num_nodes']}")


if __name__ == "__main__":
    main()
