# ---------------------------------------------------------------------------
# Amazon S3 Vectors（ベクトルストア / OpenSearch Serverless の低コスト代替）
#
# OpenSearch Serverless は最小 OCU の常時課金が効くが、S3 Vectors は
# ストレージ + リクエスト課金で常時コンピュート課金が無い（最大90%減）。
# 低QPS・コスト重視の RAG 用途に向く。Bedrock KB にネイティブ統合。
# ---------------------------------------------------------------------------

resource "aws_s3vectors_vector_bucket" "this" {
  vector_bucket_name = "${var.name_prefix}-vectors-${local.account_id}"
  force_destroy      = true
}

resource "aws_s3vectors_index" "this" {
  vector_bucket_name = aws_s3vectors_vector_bucket.this.vector_bucket_name
  index_name         = var.vector_index_name
  data_type          = "float32"
  dimension          = var.embedding_dimension # Titan Text Embeddings v2 = 1024
  distance_metric    = "cosine"

  # Bedrock KB はチャンク本文を AMAZON_BEDROCK_TEXT、ソース情報を
  # AMAZON_BEDROCK_METADATA に格納する。これらは容量が大きく検索条件に
  # 使わないため non-filterable にする（filterable メタデータのサイズ上限回避）。
  metadata_configuration {
    non_filterable_metadata_keys = [
      "AMAZON_BEDROCK_TEXT",
      "AMAZON_BEDROCK_METADATA",
    ]
  }
}
