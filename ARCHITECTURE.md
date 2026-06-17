# State-Aware RAG — AWS 再現アーキテクチャ

論文「Reasoning with Memory: Adaptive Information Management for Retrieval-Augmented Generation」(Man et al., AWS AI) を **AWS 上で実際に動かす**ための構成。

---

## 1. 論文の要点（再現対象）

| 論文の構成要素 | 役割 | 学習 |
|---|---|---|
| **Retriever** R | クエリから候補文書を取得 | 凍結 |
| **Generator** G | サブ質問・中間回答・最終回答を生成 | 凍結 |
| **Extractor** E | working memory を **filter / consolidate / update** | **学習対象** |
| **Working Memory** M_i | 推論に関連する情報のみを保持する動的な記憶（受動的な文書蓄積ではない） | — |

- 推論モードは 2 つ: **Socratic Planning**(A1+A5, 効率重視) と **MCTS**(A1〜A5, 網羅探索, globally-shared memory)
- 学習は **Path-Outcome Dual Reward**（GRPO）。ただし論文 Table 3 では **「Claude をそのまま Extractor に使う」構成が最高性能** → **学習なしで AWS Bedrock 上に再現可能**。

### Extractor の 2 フェーズ（論文 Eq.1, 2）
```
(1) Consolidation:  I_i = E_consolidate(q_i, D_i, M_{i-1})   # 関連抽出・矛盾解消・重複排除
(2) Memory Update:  M_i = E_update(x, M_{i-1}, I_i, (q_i, r_i))
```

---

## 2. AWS マッピング

| 論文の実装 | AWS での再現 |
|---|---|
| Generator = Qwen3-8B (SGLang) | **Amazon Bedrock** Claude Sonnet 4.5 |
| Extractor = Qwen3-4B (学習済み) | **Amazon Bedrock** Claude（prompt-only / 学習不要）※ 学習版は SageMaker にも差替可 |
| Judge = Claude 3.7 Sonnet | **Amazon Bedrock** Claude 3.7 Sonnet |
| Retriever = Qwen3-Embedding-4B + FAISS | **Bedrock Knowledge Base** (S3 Vectors + Titan Embeddings v2) ※常時課金回避。OpenSearch Serverless にも切替可 |
| LiteLLM プロキシ / ロードバランス | Bedrock 側でマネージド（adaptive retry） |
| Wikipedia 2023 dump | **S3** に格納し KB のデータソースに |

---

## 3. アーキテクチャ図

```
                                 ┌──────────────────────────────────────────┐
                                 │            State-Aware RAG App             │
                                 │   (ローカル / Lambda / ECS Fargate)         │
   ┌──────────┐   question       │                                            │
   │  User /  │ ───────────────► │   pipeline.py  ──►  Socratic / MCTS planner│
   │  Client  │                  │        │                                   │
   └──────────┘ ◄─────────────── │        │  ┌─────────── Working Memory ───┐ │
                  final answer    │        │  │ important_info + QA history  │ │
                                 │        │  └──────────────────────────────┘ │
                                 │        ▼                                   │
                                 │  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
                                 │  │Generator│  │Extractor │  │  Judge    │  │
                                 │  │  (G)    │  │   (E)    │  │ (reward)  │  │
                                 │  └────┬────┘  └────┬─────┘  └─────┬─────┘  │
                                 └───────┼────────────┼──────────────┼────────┘
                                         │            │              │
                  ┌──────────────────────┴────────────┴──────────────┴───────┐
                  │              Amazon Bedrock (Converse API)                │
                  │   Claude Sonnet 4.5 (global 推論プロファイル) / Titan v2   │
                  └───────────────────────────────────────────────────────────┘
                                         │ retrieve(query)
                                         ▼
                  ┌───────────────────────────────────────────────────────────┐
                  │           Amazon Bedrock Knowledge Base                     │
                  │   ┌──────────────┐        ┌──────────────────────────────┐ │
                  │   │  S3 (corpus) │ ─sync─► │ S3 Vectors                   │ │
                  │   │ Wikipedia等  │        │ (vector index / cosine, 1024)│ │
                  │   └──────────────┘        └──────────────────────────────┘ │
                  └───────────────────────────────────────────────────────────┘
```

### 推論シーケンス（Socratic, 1 ステップ）
```
1. Generator: サブ質問 q_i を生成（memory で答えられるか判定）
2. Generator: 検索クエリ生成 → Retriever(KB) で文書 D_i 取得
3. Extractor: Consolidation  I_i = E(q_i, D_i, M_{i-1})        … Eq.1
4. Generator: 中間回答 r_i = G(q_i, I_i + M_{i-1})
5. Extractor: Memory Update  M_i = E(x, M_{i-1}, I_i, (q_i,r_i)) … Eq.2
   → 答えられるようになるまで繰り返し → A5 Conclude で最終回答
```

---

## 4. デプロイ構成の選択肢

| 用途 | Retriever | LLM | 実行環境 |
|---|---|---|---|
| **ローカル検証** | FAISS (`SAR_RETRIEVER_BACKEND=faiss`) | Bedrock | ローカル Python |
| **本番（推奨）** | Bedrock KB (`bedrock_kb`) | Bedrock | Lambda / ECS Fargate |
| **学習版 Extractor** | Bedrock KB | Generator=Bedrock, Extractor=SageMaker エンドポイント | ECS Fargate |

学習版 Extractor を使う場合は、論文 Appendix B の手順（Axolotl で SFT → verl + vLLM で GRPO、Qwen3-4B-Thinking ベース、λ=2）で学習し、SageMaker にデプロイして `SAR_EXTRACTOR_MODEL` をそのエンドポイントに向ける。本リポジトリの推論パイプラインはそのまま流用できる（plug-and-play）。

---

## 5. コスト/性能メモ（論文 Section 5.3）

- 推論ステップは **5** で飽和（Bamboogle）。
- MCTS は Socratic より **約 6.8 倍**の Generator 呼び出しで +2.6pt 程度 → 多くの用途では Socratic が費用対効果良。
- `SAR_MCTS_ROLLOUTS` を増やすほど MCTS は改善（3→15 で +3.3pt）。
- 既定値は論文 Table 7 準拠（max_depth=5, rollouts=10, top_k=5, exploration_weight=1.0）。
