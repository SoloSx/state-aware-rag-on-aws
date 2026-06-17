# State-Aware RAG on AWS

論文 **"Reasoning with Memory: Adaptive Information Management for Retrieval-Augmented Generation"** (Man et al., AWS AI) を **AWS Bedrock 上で実際に再現**する実装。

- 推論モード: **Socratic Planning** / **MCTS**（論文 Section 3.3）
- 全プロンプトは論文 Appendix C を完全転記（`state_aware_rag/prompts.py`）
- Extractor は **学習不要**（論文 Table 3 で Claude をそのまま Extractor に使う構成が最高性能）
- 学習版 Qwen3-4B Extractor に差し替えるのも 1 行（`SAR_EXTRACTOR_MODEL`）

詳細な構成は [ARCHITECTURE.md](./ARCHITECTURE.md) を参照。

---

## ディレクトリ構成

```
state_aware_rag/
  config.py       設定（環境変数で上書き / 論文 Table 7 準拠の既定値）
  prompts.py      論文 Appendix C の全プロンプト
  llm.py          Bedrock Converse API ラッパー（Generator/Extractor/Judge）
  retriever.py    Retriever（Bedrock KB / ローカル FAISS）
  memory.py       Working Memory（論文の中核貢献）
  extractor.py    Extractor: Consolidation(Eq.1) + Memory Update(Eq.2)
  generator.py    Generator: A1〜A5 の各操作
  judge.py        LLM-as-a-Judge（Path/Outcome 報酬・評価指標）
  socratic.py     Socratic Planner（A1+A5）
  mcts.py         MCTS Planner（A1〜A5 + globally-shared memory）
  pipeline.py     高レベル API
  cli.py          CLI
ingest/           FAISS インデックス構築（ローカル検証用）
infra/terraform/  Terraform（S3 + S3 Vectors + Bedrock KB / 東京リージョン）
eval/             定量評価ハーネス（Sub-EM / Acc / RAGAS）
examples/         動作例 + サンプルコーパス
```

---

## クイックスタート

### 前提
1. AWS 認証情報（`aws configure` 済み）
2. **Bedrock モデルアクセスを有効化**（コンソール → Bedrock → Model access）:
   - `anthropic.claude-sonnet-4-5` / `anthropic.claude-3-7-sonnet`
   - `amazon.titan-embed-text-v2:0`

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### A. ローカル検証（FAISS, 最速）

```bash
export AWS_REGION=ap-northeast-1
export AWS_PROFILE=your-aws-profile
export SAR_RETRIEVER_BACKEND=faiss

# サンプルコーパスから FAISS インデックスを構築
python ingest/build_faiss_index.py \
  --input examples/sample_corpus.jsonl \
  --index data/faiss.index --docs data/docs.jsonl

# 実行
python -m state_aware_rag.cli --mode socratic -v \
  "Are the director of film Move (1970) and the director of film Mediterranee (1963) from the same country?"

# まとめて実行する例
python examples/run_example.py
```

### B. 本番（Bedrock Knowledge Base）

```bash
# 1) インフラをデプロイ（Terraform / 東京リージョン / your-aws-profile）
cd infra/terraform
terraform init
terraform apply      # 出力に knowledge_base_id / corpus_bucket / data_source_id

# 2) コーパスを S3 に投入して KB を同期
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
python -m state_aware_rag.cli --mode mcts --rollouts 10 -v "your multi-hop question"
```

---

## Python から使う

```python
from state_aware_rag import StateAwareRAG

rag = StateAwareRAG()
result = rag.answer("your question", mode="socratic")  # or "mcts"
print(result.answer)
print(result.details)   # steps / memory / scores

# ベンチマーク評価（論文の Acc 指標, LLM-as-a-Judge）
ok = rag.evaluate(question, result.answer, gold_answer)
```

---

## 主な設定（環境変数）

| 変数 | 既定 | 説明 |
|---|---|---|
| `SAR_MODE` | `socratic` | `socratic` / `mcts` |
| `SAR_MAX_STEPS` | `5` | 最大推論ステップ（論文の飽和点） |
| `SAR_MCTS_ROLLOUTS` | `10` | MCTS rollout 数（Table 7） |
| `SAR_RETRIEVER_BACKEND` | `bedrock_kb` | `bedrock_kb` / `faiss` / `web`（DuckDuckGo, 鍵不要） |
| `SAR_KB_ID` | — | Bedrock Knowledge Base ID |
| `AWS_REGION` | `ap-northeast-1` | デプロイ/推論リージョン（東京） |
| `SAR_GENERATOR_MODEL` | `global.anthropic.claude-sonnet-4-5...` | Generator（global 推論プロファイル。geo比 約10%安い） |
| `SAR_EXTRACTOR_MODEL` | `global.anthropic.claude-sonnet-4-5...` | Extractor（SageMaker 学習版に差替可） |
| `SAR_JUDGE_MODEL` | `global.anthropic.claude-sonnet-4-5...` | Judge（データ所在要件があれば `jp.` に変更） |

---

## 定量評価（Sub-EM / Acc / RAGAS）

```bash
pip install -r eval/requirements.txt

# 自作データセット（KB の S3 Vectors を使う）
python -m eval.run_eval --mode socratic --ragas

# 反復メモリの貢献を切り分けるアブレーション（論文 Table 2 相当）
# 架空エンティティの多段QAで、メモリ ON/OFF を比較
python ingest/build_faiss_index.py --input eval/synthetic_corpus.jsonl \
  --index data/synth.index --docs data/synth_docs.jsonl
SAR_RETRIEVER_BACKEND=faiss SAR_FAISS_INDEX=data/synth.index SAR_FAISS_DOCS=data/synth_docs.jsonl \
  SAR_ABLATION=full python -m eval.run_eval --dataset eval/synthetic_eval.jsonl
# → full: Acc 100% / no_memory_update: Acc 0%（反復メモリが必須なことを実証）

# 論文と同じ Bamboogle を Web 検索リトリーバで実ベンチ
SAR_RETRIEVER_BACKEND=web \
  python -m eval.run_eval --mode socratic \
  --dataset eval/bamboogle.jsonl --limit 10
```

- 論文指標 **Sub-EM**（部分一致）/ **Acc**（LLM-judge）と **RAGAS** 5指標
- `SAR_ABLATION`: `full` / `no_memory_update` / `no_consolidation` / `no_extractor`（反復メモリの切り分け）
- `web` バックエンドは論文 Component Analysis の Google Search 変種に相当（DuckDuckGo、鍵不要）

---

## 論文との対応

| 論文 | 本実装 |
|---|---|
| Eq.1 Consolidation | `extractor.Extractor.consolidate` |
| Eq.2 Memory Update | `extractor.Extractor.update` |
| Eq.3 Path Reward | `judge.Judge.path_reward` |
| Eq.4 Outcome Reward | `judge.Judge.outcome_reward` |
| A1 Decompose&Answer | `socratic` / `mcts._apply_action("A1")` |
| A2〜A5 | `mcts._apply_action` |
| globally-shared memory | `mcts.MCTSPlanner.shared_memory` + `_merge_shared` |
| Table 7 パラメータ | `config.InferenceConfig` |
| Appendix C プロンプト | `prompts.py` |

> 注: 本リポジトリは論文の**推論フレームワークの再現**に焦点を当てています。Extractor の RL 学習（GRPO）は計算資源を要するため任意で、学習済みモデルは `SAR_EXTRACTOR_MODEL` で差し替えられます。論文自身が「Claude を Extractor に使う」構成を最高性能として報告しているため、学習なしでも論文の本質的な仕組み（動的 working memory 管理）を再現できます。
