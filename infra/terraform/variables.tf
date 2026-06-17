variable "region" {
  description = "デプロイ先 AWS リージョン"
  type        = string
  default     = "ap-northeast-1" # 東京
}

variable "aws_profile" {
  description = "使用する AWS CLI プロファイル"
  type        = string
  default     = "default"
}

variable "environment" {
  description = "環境名（タグ用）"
  type        = string
  default     = "dev"
}

variable "name_prefix" {
  description = "リソース名のプレフィックス"
  type        = string
  default     = "state-aware-rag"
}

variable "embedding_model_id" {
  description = "Bedrock 埋め込みモデル ID（KB 用）"
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}

variable "embedding_dimension" {
  description = "埋め込みベクトルの次元数（Titan Text Embeddings v2 = 1024）"
  type        = number
  default     = 1024
}

variable "vector_index_name" {
  description = "S3 Vectors のベクトルインデックス名"
  type        = string
  default     = "sar-index"
}
