# ---------------------------------------------------------------------------
# Amazon Bedrock Knowledge Base（論文の Retriever Server に相当）
# ---------------------------------------------------------------------------
resource "aws_bedrockagent_knowledge_base" "this" {
  name     = "${var.name_prefix}-kb"
  role_arn = aws_iam_role.knowledge_base.arn

  knowledge_base_configuration {
    type = "VECTOR"
    vector_knowledge_base_configuration {
      embedding_model_arn = local.embedding_model_arn
    }
  }

  storage_configuration {
    type = "S3_VECTORS"
    s3_vectors_configuration {
      # index_arn 単体でインデックス（バケット含む）を特定する。
      # vector_bucket_arn との併用は不可。
      index_arn = aws_s3vectors_index.this.index_arn
    }
  }

  # インデックスと権限が整ってから作成する
  depends_on = [
    aws_s3vectors_index.this,
    aws_iam_role_policy.knowledge_base,
  ]
}

# S3 データソース（コーパス）
resource "aws_bedrockagent_data_source" "corpus" {
  knowledge_base_id = aws_bedrockagent_knowledge_base.this.id
  name              = "s3-corpus"

  data_source_configuration {
    type = "S3"
    s3_configuration {
      bucket_arn = aws_s3_bucket.corpus.arn
    }
  }

  # 文書削除時に KB からも除去（best practice）
  data_deletion_policy = "RETAIN"
}
