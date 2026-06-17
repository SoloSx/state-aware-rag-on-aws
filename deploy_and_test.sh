#!/usr/bin/env bash
# State-Aware RAG を東京リージョン(your-aws-profile)に実際にデプロイし、
# サンプルコーパスで end-to-end 動作確認まで一気に行うスクリプト。
#
# 前提: `aws login --profile $AWS_PROFILE` 済み（セッション有効）
#
# 使い方:
#   ./deploy_and_test.sh
set -euo pipefail

export AWS_PROFILE=your-aws-profile
export AWS_REGION=ap-northeast-1
ROOT="$(cd "$(dirname "$0")" && pwd)"
TF_DIR="$ROOT/infra/terraform"

green() { printf "\033[32m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

# ---------------------------------------------------------------------------
# 0. 認証・モデルアクセスのプリフライト
# ---------------------------------------------------------------------------
green "== [0/5] 認証チェック =="
if ! aws sts get-caller-identity >/dev/null 2>&1; then
  red "your-aws-profile のセッションが無効です。先に 'aws login --profile $AWS_PROFILE' を実行してください。"
  exit 1
fi
aws sts get-caller-identity --output table

green "== Bedrock モデルアクセス（global 推論プロファイル）の確認 =="
for mid in "global.anthropic.claude-sonnet-4-5-20250929-v1:0"; do
  if aws bedrock get-inference-profile --inference-profile-identifier "$mid" >/dev/null 2>&1; then
    green "  OK: $mid"
  else
    red "  注意: $mid にアクセスできません。Bedrock コンソールでモデルアクセスを有効化してください。"
  fi
done

# ---------------------------------------------------------------------------
# 1. Terraform でインフラを作成
# ---------------------------------------------------------------------------
green "== [1/5] terraform apply =="
cd "$TF_DIR"
# S3 Vectors 構成では opensearch プロバイダが不要なので、aws プロバイダが
# プロファイル（your-aws-profile）を直接使える。env クレデンシャル展開は不要。
terraform init -input=false >/dev/null
terraform apply -auto-approve

KB_ID="$(terraform output -raw knowledge_base_id)"
DS_ID="$(terraform output -raw data_source_id)"
BUCKET="$(terraform output -raw corpus_bucket)"
green "  KnowledgeBase=$KB_ID  DataSource=$DS_ID  Bucket=$BUCKET"

# ---------------------------------------------------------------------------
# 2. サンプルコーパスを 1 パッセージ 1 ファイルにして S3 へ投入
# ---------------------------------------------------------------------------
green "== [2/5] サンプルコーパスを S3 へアップロード =="
TMP="$(mktemp -d)"
python3 - "$ROOT/examples/sample_corpus.jsonl" "$TMP" <<'PY'
import json, sys, pathlib
src, out = sys.argv[1], pathlib.Path(sys.argv[2])
for line in open(src, encoding="utf-8"):
    rec = json.loads(line)
    (out / f"{rec['id']}.txt").write_text(rec["text"], encoding="utf-8")
print("wrote", len(list(out.glob('*.txt'))), "files")
PY
aws s3 cp "$TMP/" "s3://$BUCKET/" --recursive
rm -rf "$TMP"

# ---------------------------------------------------------------------------
# 3. Knowledge Base への取り込み（ingestion job）
# ---------------------------------------------------------------------------
green "== [3/5] ingestion job 開始 =="
JOB_ID="$(aws bedrock-agent start-ingestion-job \
  --knowledge-base-id "$KB_ID" --data-source-id "$DS_ID" \
  --query 'ingestionJob.ingestionJobId' --output text)"
green "  job=$JOB_ID 完了待ち..."
while true; do
  STATUS="$(aws bedrock-agent get-ingestion-job \
    --knowledge-base-id "$KB_ID" --data-source-id "$DS_ID" \
    --ingestion-job-id "$JOB_ID" --query 'ingestionJob.status' --output text)"
  echo "    status=$STATUS"
  [ "$STATUS" = "COMPLETE" ] && break
  [ "$STATUS" = "FAILED" ] && { red "ingestion failed"; exit 1; }
  sleep 10
done

# ---------------------------------------------------------------------------
# 4. Python 依存のセットアップ
# ---------------------------------------------------------------------------
green "== [4/5] Python ランタイム準備 =="
cd "$ROOT"
if [ ! -d .venv ]; then python3.11 -m venv .venv 2>/dev/null || python3 -m venv .venv; fi
# botocore[crt] を入れると your-aws-profile の login プロファイルを boto3 が直接使え、
# 長時間実行でもセッションが自動更新される（env 展開だと途中で失効する）。
./.venv/bin/pip install -q "boto3" "botocore[crt]"

# ---------------------------------------------------------------------------
# 5. 実際に推論を実行（Bedrock KB バックエンド）
# ---------------------------------------------------------------------------
green "== [5/5] State-Aware RAG を実行 =="
export SAR_RETRIEVER_BACKEND=bedrock_kb
export SAR_KB_ID="$KB_ID"

# botocore[crt] 済みなので AWS_PROFILE を直接使える（セッション自動更新）。
RUN=(./.venv/bin/python -m state_aware_rag.cli)

Q="Are the director of film Move (1970) and the director of film Mediterranee (1963) from the same country?"
green "-- Socratic --"
"${RUN[@]}" --mode socratic -v "$Q"
green "-- MCTS (rollouts=5) --"
"${RUN[@]}" --mode mcts --rollouts 5 -v "$Q"

green "== 完了。SAR_KB_ID=$KB_ID =="
