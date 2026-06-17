# State-Aware RAG on AWS

論文 **"Reasoning with Memory: Adaptive Information Management for Retrieval-Augmented Generation"** (Man et al., Amazon AI) を **Amazon Bedrock + Amazon S3 Vectors** で動かす実装と、定量評価 (Sub-EM / Acc / RAGAS) のハーネス。

- 推論モード: **Socratic Planning** / **MCTS** (論文 Section 3.3)
- 全プロンプトは論文 Appendix C を完全転記 (`state_aware_rag/prompts.py`)
- Extractor は **学習不要** (論文 Table 3 で Claude をそのまま Extractor に使う構成が最高性能)
- 学習版 Qwen3-4B Extractor に差し替えるのも 1 行 (`SAR_EXTRACTOR_MODEL`)

詳細な構成は [ARCHITECTURE.md](./ARCHITECTURE.md) を参照。

---

## ディレクトリ構成

```text
state-aware-rag/
├── pyproject.toml           # 依存定義 (uv で管理)
├── uv.lock
├── state_aware_rag/         # 推論フレームワーク本体
│   ├── prompts.py           # 論文 Appendix C の全プロンプト
│   ├── llm.py               # Amazon Bedrock Converse API ラッパー
│   ├── retriever.py         # Amazon Bedrock Knowledge Bases / ローカル FAISS
│   ├── extractor.py         # Consolidation(Eq.1) + Memory Update(Eq.2)
│   ├── generator.py         # Generator (A1〜A5 の各操作)
│   ├── judge.py             # LLM-as-a-Judge (Path/Outcome 報酬・評価指標)
│   ├── memory.py            # ワーキングメモリの状態管理
│   ├── socratic.py          # Socratic Planner (A1+A5)
│   ├── mcts.py              # MCTS Planner (A1-A5 + globally-shared memory)
│   ├── config.py            # モデル ID やリトリーバ設定 (論文 Table 7 準拠)
│   ├── pipeline.py          # 高レベル API
│   └── cli.py               # CLI
├── infra/terraform/         # Amazon S3 + Amazon S3 Vectors + Amazon Bedrock Knowledge Bases
├── ingest/                  # FAISS インデックス構築 (ローカル検証用)
├── examples/                # 動作例 + サンプルコーパス
└── eval/                    # 定量評価ハーネス (Sub-EM / Acc / RAGAS)
```

---

## クイックスタート

### 前提
1. AWS 認証情報 (`aws configure` 済み、東京リージョン)
2. Amazon Bedrock のモデルアクセスを有効化 (コンソール → Amazon Bedrock → Model access):
   - `global.anthropic.claude-sonnet-4-5-20250929-v1:0`
   - `amazon.titan-embed-text-v2:0`
3. Python 3.11+ と [uv](https://github.com/astral-sh/uv)

```bash
uv sync --extra eval      # pyproject.toml + uv.lock から依存をインストール (評価用も込み)
```

### A. ローカル検証 (FAISS, 最速)

```bash
export AWS_REGION=ap-northeast-1
export AWS_PROFILE=your-aws-profile
export SAR_RETRIEVER_BACKEND=faiss

# サンプルコーパスから FAISS インデックスを構築
uv run python ingest/build_faiss_index.py \
  --input examples/sample_corpus.jsonl \
  --index data/faiss.index --docs data/docs.jsonl

# 実行
uv run python -m state_aware_rag.cli --mode socratic -v \
  "Are the director of film Move (1970) and the director of film Mediterranee (1963) from the same country?"
```

### B. 本番 (Amazon Bedrock Knowledge Bases + Amazon S3 Vectors)

```bash
# 1) インフラをデプロイ
cd infra/terraform
terraform init
terraform apply      # 出力に knowledge_base_id / corpus_bucket / data_source_id

# 2) コーパスを Amazon S3 に投入して Knowledge Bases を同期
aws s3 cp ../../corpus/ s3://$(terraform output -raw corpus_bucket)/ --recursive --profile $AWS_PROFILE
aws bedrock-agent start-ingestion-job \
  --knowledge-base-id $(terraform output -raw knowledge_base_id) \
  --data-source-id    $(terraform output -raw data_source_id) \
  --region ap-northeast-1 --profile $AWS_PROFILE

# 3) 推論
cd ../..
export AWS_REGION=ap-northeast-1
export AWS_PROFILE=your-aws-profile
export SAR_RETRIEVER_BACKEND=bedrock_kb
export SAR_KB_ID=$(cd infra/terraform && terraform output -raw knowledge_base_id)
uv run python -m state_aware_rag.cli --mode mcts --rollouts 10 -v "your multi-hop question"
```

---

## Python から使う

```python
from state_aware_rag import StateAwareRAG

rag = StateAwareRAG()
result = rag.answer("your question", mode="socratic")  # or "mcts"
print(result.answer)
print(result.details)   # steps / memory / scores

# ベンチマーク評価 (論文の Acc 指標、LLM-as-a-Judge)
ok = rag.evaluate(question, result.answer, gold_answer)
```

---

## 主な設定 (環境変数)

| 変数 | 既定 | 説明 |
|---|---|---|
| `SAR_MODE` | `socratic` | `socratic` / `mcts` |
| `SAR_MAX_STEPS` | `5` | 最大推論ステップ (論文の飽和点) |
| `SAR_MCTS_ROLLOUTS` | `10` | MCTS rollout 数 (Table 7) |
| `SAR_RETRIEVER_BACKEND` | `bedrock_kb` | `bedrock_kb` / `faiss` / `web` (DuckDuckGo, 鍵不要) |
| `SAR_KB_ID` | — | Amazon Bedrock Knowledge Bases の ID |
| `SAR_ABLATION` | `full` | `full` / `no_memory_update` / `no_consolidation` / `no_extractor` |
| `AWS_REGION` | `ap-northeast-1` | デプロイ/推論リージョン (東京) |
| `SAR_GENERATOR_MODEL` | `global.anthropic.claude-sonnet-4-5-20250929-v1:0` | Generator |
| `SAR_EXTRACTOR_MODEL` | `global.anthropic.claude-sonnet-4-5-20250929-v1:0` | Extractor (SageMaker 学習版に差替可) |
| `SAR_JUDGE_MODEL` | `global.anthropic.claude-sonnet-4-5-20250929-v1:0` | Judge (データ所在要件があれば `jp.` に変更) |

---

## 定量評価 (Sub-EM / Acc / RAGAS)

```bash
# 自作データセット (Amazon Bedrock Knowledge Bases + Amazon S3 Vectors を使う)
uv run python -m eval.run_eval --mode socratic --ragas

# 反復メモリの貢献を切り分けるアブレーション (論文 Table 2 相当)
# 架空エンティティの多段 QA で、メモリ ON/OFF を比較
uv run python ingest/build_faiss_index.py --input eval/synthetic_corpus.jsonl \
  --index data/synth.index --docs data/synth_docs.jsonl
SAR_RETRIEVER_BACKEND=faiss SAR_FAISS_INDEX=data/synth.index SAR_FAISS_DOCS=data/synth_docs.jsonl \
  SAR_ABLATION=full uv run python -m eval.run_eval --dataset eval/synthetic_eval.jsonl
# → full: Acc 100% / no_memory_update: Acc 0% (反復メモリが必須なことを実証)

# 論文と同じ Bamboogle を Web 検索リトリーバで実ベンチ
SAR_RETRIEVER_BACKEND=web \
  uv run python -m eval.run_eval --mode socratic \
  --dataset eval/bamboogle.jsonl --limit 10
```

- 論文指標 **Sub-EM** (部分一致) / **Acc** (LLM-judge) と **RAGAS** 5指標
- `web` バックエンドは論文 Component Analysis の Google Search 変種に相当 (DuckDuckGo、鍵不要)

---

## 論文との対応

| 論文 | 本実装 |
|---|---|
| Eq.1 Consolidation | `extractor.Extractor.consolidate` |
| Eq.2 Memory Update | `extractor.Extractor.update` |
| Eq.3 Path Reward | `judge.Judge.path_reward` |
| Eq.4 Outcome Reward | `judge.Judge.outcome_reward` |
| A1 Decompose & Answer | `socratic` / `mcts._apply_action("A1")` |
| A2-A5 | `mcts._apply_action` |
| globally-shared memory | `mcts.MCTSPlanner.shared_memory` + `_merge_shared` |
| Table 7 パラメータ | `config.InferenceConfig` |
| Appendix C プロンプト | `prompts.py` |

> 注: 本リポジトリは論文の**推論フレームワークの再現**に焦点を当てています。Extractor の RL 学習 (GRPO) は計算資源を要するため任意で、学習済みモデルは `SAR_EXTRACTOR_MODEL` で差し替えられます。論文自身が「Claude を Extractor に使う」構成を最高性能として報告しているため、学習なしでも論文の本質的な仕組み (動的なワーキングメモリ管理) を再現できます。
