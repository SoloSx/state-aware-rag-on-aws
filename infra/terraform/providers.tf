provider "aws" {
  region  = var.region
  profile = var.aws_profile

  default_tags {
    tags = {
      Project   = "state-aware-rag"
      ManagedBy = "terraform"
      Env       = var.environment
    }
  }
}

# 現在の呼び出し元・リージョン・パーティションを取得
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
data "aws_partition" "current" {}

# 補足: S3 Vectors はインデックスを aws_s3vectors_index で直接作成できるため、
# OpenSearch Serverless 構成のような opensearch プロバイダや SigV4 署名・
# クレデンシャルの env 展開は不要になった。
