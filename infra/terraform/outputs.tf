output "knowledge_base_id" {
  description = "Bedrock Knowledge Base ID（SAR_KB_ID に設定）"
  value       = aws_bedrockagent_knowledge_base.this.id
}

output "data_source_id" {
  description = "S3 データソース ID（ingestion job で使用）"
  value       = aws_bedrockagent_data_source.corpus.data_source_id
}

output "corpus_bucket" {
  description = "コーパス文書をアップロードする S3 バケット名"
  value       = aws_s3_bucket.corpus.bucket
}

output "vector_bucket_name" {
  description = "S3 Vectors ベクトルバケット名"
  value       = aws_s3vectors_vector_bucket.this.vector_bucket_name
}

output "vector_index_arn" {
  description = "S3 Vectors ベクトルインデックスの ARN"
  value       = aws_s3vectors_index.this.index_arn
}

output "kb_role_arn" {
  description = "Knowledge Base 実行ロールの ARN"
  value       = aws_iam_role.knowledge_base.arn
}

output "usage_hint" {
  description = "アプリ実行用の環境変数"
  value       = <<-EOT
    export AWS_REGION=${var.region}
    export AWS_PROFILE=${var.aws_profile}
    export SAR_RETRIEVER_BACKEND=bedrock_kb
    export SAR_KB_ID=${aws_bedrockagent_knowledge_base.this.id}
  EOT
}
