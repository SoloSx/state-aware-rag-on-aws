# State-Aware RAG — Terraform インフラ

論文の Retriever Server (Amazon Bedrock Knowledge Bases + Amazon S3 Vectors + Amazon Titan Embeddings v2) を東京リージョン (ap-northeast-1) にデプロイする。

## 構成リソース
- `aws_s3_bucket` — コーパス文書 (Knowledge Bases のデータソース)
- `aws_s3vectors_vector_bucket` / `aws_s3vectors_index` — ベクトルストア (cosine, 1024次元)
- `aws_bedrockagent_knowledge_base` + `aws_bedrockagent_data_source`
- `aws_iam_role` — Knowledge Bases 実行ロール (最小権限 + confused deputy 対策)

## プロバイダ
- `hashicorp/aws ~> 6.0`

## 前提
1. Amazon Bedrock のモデルアクセスを有効化 (東京リージョン):
   - `amazon.titan-embed-text-v2:0`
   - `global.anthropic.claude-sonnet-4-5-20250929-v1:0`
2. お使いの AWS プロファイルでログイン済みであること:
   ```bash
   aws sts get-caller-identity --profile $AWS_PROFILE
   ```

## デプロイ
```bash
cd infra/terraform
terraform init
terraform plan
terraform apply
```

## デプロイ後
```bash
# 出力から環境変数を設定
terraform output -raw usage_hint

# コーパスを S3 に投入
aws s3 cp ../../corpus/ s3://$(terraform output -raw corpus_bucket)/ --recursive --profile $AWS_PROFILE

# Knowledge Base に取り込み (ingestion job)
aws bedrock-agent start-ingestion-job \
  --knowledge-base-id $(terraform output -raw knowledge_base_id) \
  --data-source-id    $(terraform output -raw data_source_id) \
  --region ap-northeast-1 --profile $AWS_PROFILE
```

## 破棄
```bash
terraform destroy
```

## 注意点
- Amazon S3 Vectors は AWS provider 6.x 系で対応。
- Amazon S3 Vectors はストレージ + リクエスト課金で、Amazon OpenSearch Serverless のような最小 OCU の常時課金は発生しません。
- 本番運用ではリモートステート (S3 + DynamoDB ロック) を推奨。`versions.tf` のコメント参照。
