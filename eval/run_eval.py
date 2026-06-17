"""State-Aware RAG の定量評価ハーネス。

論文準拠の指標（Sub-EM / Acc）に加え、RAGAS による RAG 特化指標
（faithfulness / answer_relevancy / context_precision / context_recall /
answer_correctness）を計算する。LLM・埋め込みはすべて Amazon Bedrock を使う。

使い方:
    # 論文指標のみ（追加依存なし）
    python -m eval.run_eval --mode socratic

    # RAGAS も含める（pip install -r eval/requirements.txt が必要）
    python -m eval.run_eval --mode socratic --ragas

    # 件数を絞る（コスト配慮）
    python -m eval.run_eval --mode socratic --limit 2 --ragas

出力:
    eval/results_<mode>.json … 1問ごとの詳細
    標準出力 … 集計サマリ
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# パッケージ解決（リポジトリルートを import パスに追加）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from eval import paper_metrics
from state_aware_rag import StateAwareRAG
from state_aware_rag.config import load_config

DATA = pathlib.Path(__file__).parent / "eval_dataset.jsonl"


def load_dataset(limit: int | None, path: str | None = None) -> list[dict]:
    p = pathlib.Path(path) if path else DATA
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def run_inference(rag: StateAwareRAG, dataset: list[dict], mode: str) -> list[dict]:
    """各問に推論を実行し、評価に必要なフィールドを集める。"""
    results = []
    for i, item in enumerate(dataset, 1):
        q = item["question"]
        print(f"  [{i}/{len(dataset)}] {q[:60]}...", file=sys.stderr)
        ans = rag.answer(q, mode=mode)
        contexts = ans.details.get("retrieved_contexts", [])
        results.append(
            {
                "question": q,
                "answers": item.get("answers", []),
                "ground_truth": item.get("ground_truth", ""),
                "prediction": ans.answer,
                "retrieved_contexts": contexts,
                # 論文指標
                "sub_em": paper_metrics.sub_em(ans.answer, item.get("answers", [])),
                "acc": rag.evaluate(q, ans.answer, item.get("ground_truth", "")),
            }
        )
    return results


def run_ragas(results: list[dict], cfg) -> dict:
    """RAGAS で RAG 特化指標を計算（Bedrock を LLM/埋め込みに使用）。"""
    try:
        from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_correctness,
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        )
    except ImportError as e:
        print(f"\n[RAGAS skip] 依存が未インストールです: {e}\n  pip install -r eval/requirements.txt", file=sys.stderr)
        return {}

    # temperature のみ指定（Sonnet 4.5 は temperature と top_p の同時指定不可）
    judge = ChatBedrockConverse(
        model=cfg.bedrock.judge_model_id, region_name=cfg.bedrock.region, temperature=0
    )
    embeddings = BedrockEmbeddings(
        model_id=cfg.retriever.embedding_model_id, region_name=cfg.bedrock.region
    )
    ragas_llm = LangchainLLMWrapper(judge)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)

    samples = [
        {
            "user_input": r["question"],
            "response": r["prediction"],
            "retrieved_contexts": r["retrieved_contexts"] or ["(no context retrieved)"],
            "reference": r["ground_truth"],
        }
        for r in results
    ]
    dataset = EvaluationDataset.from_list(samples)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]
    scores = evaluate(dataset=dataset, metrics=metrics, llm=ragas_llm, embeddings=ragas_emb)
    # ragas EvaluationResult -> 指標ごとの平均（公開API to_pandas を使用）
    df = scores.to_pandas()
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]
    return {c: round(float(df[c].mean()), 3) for c in metric_cols if c in df.columns}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="State-Aware RAG 定量評価")
    ap.add_argument("--mode", choices=["socratic", "mcts"], default="socratic")
    ap.add_argument("--limit", type=int, default=None, help="評価件数の上限（コスト配慮）")
    ap.add_argument("--ragas", action="store_true", help="RAGAS 指標も計算する")
    ap.add_argument("--dataset", default=None, help="評価データセット JSONL（既定: eval_dataset.jsonl）")
    ap.add_argument("--out", default=None, help="詳細結果の出力先 JSON")
    args = ap.parse_args(argv)

    cfg = load_config()
    rag = StateAwareRAG(cfg)
    dataset = load_dataset(args.limit, args.dataset)

    print(f"== 推論実行: mode={args.mode}, n={len(dataset)} ==", file=sys.stderr)
    results = run_inference(rag, dataset, args.mode)

    summary = {
        "mode": args.mode,
        "n": len(results),
        "paper_metrics": {
            "sub_em(%)": paper_metrics.aggregate(results, "sub_em"),
            "acc(%)": paper_metrics.aggregate(results, "acc"),
        },
    }

    if args.ragas:
        print("== RAGAS 評価 ==", file=sys.stderr)
        summary["ragas"] = run_ragas(results, cfg)

    out_path = args.out or str(pathlib.Path(__file__).parent / f"results_{args.mode}.json")
    pathlib.Path(out_path).write_text(
        json.dumps({"summary": summary, "details": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 50)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n詳細: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
