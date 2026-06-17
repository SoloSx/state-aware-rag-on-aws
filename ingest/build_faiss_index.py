"""ローカル検証用の FAISS インデックス構築スクリプト。

論文の Retriever Server（Wikipedia 2023 dump + Qwen3-Embedding-4B + FAISS）の
ローカル代替。ここでは Bedrock Titan Embeddings v2 で埋め込みを作る。

入力: JSONL（1 行 = {"id": "...", "text": "..."}）
出力: FAISS インデックス + 文書 JSONL

使い方:
    python ingest/build_faiss_index.py \
        --input data/wiki_passages.jsonl \
        --index data/faiss.index \
        --docs  data/docs.jsonl

本番（AWS）では FAISS ではなく Amazon Bedrock Knowledge Base
(OpenSearch Serverless) を使う。その場合は本スクリプト不要で、
S3 に文書を置いて KB を作成・同期するだけでよい（infra/ 参照）。
"""
from __future__ import annotations

import argparse
import json
import os

import boto3

EMBED_MODEL = os.environ.get("SAR_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
REGION = os.environ.get("AWS_REGION", "ap-northeast-1")


def embed(client, text: str) -> list[float]:
    resp = client.invoke_model(modelId=EMBED_MODEL, body=json.dumps({"inputText": text}))
    return json.loads(resp["body"].read())["embedding"]


def main() -> None:
    import faiss
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="入力 JSONL ({'id','text'})")
    ap.add_argument("--index", default="data/faiss.index")
    ap.add_argument("--docs", default="data/docs.jsonl")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.index) or ".", exist_ok=True)
    client = boto3.client("bedrock-runtime", region_name=REGION)

    vectors: list[list[float]] = []
    docs: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            vectors.append(embed(client, rec["text"]))
            docs.append({"id": rec.get("id", str(i)), "text": rec["text"]})
            if (i + 1) % 100 == 0:
                print(f"embedded {i + 1} passages")

    mat = np.array(vectors, dtype="float32")
    # cosine 類似度を内積で測るため L2 正規化
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    faiss.write_index(index, args.index)

    with open(args.docs, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"done. {len(docs)} passages -> {args.index}")


if __name__ == "__main__":
    main()
