terraform {
  required_version = ">= 1.9.0"

  required_providers {
    # 最新の AWS provider（6.x 系）。S3 Vectors / Bedrock KB を含む。
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  # 本番運用ではリモートステートを推奨（S3 + DynamoDB ロック）。
  # 例:
  # backend "s3" {
  #   bucket         = "my-tfstate"
  #   key            = "state-aware-rag/terraform.tfstate"
  #   region         = "ap-northeast-1"
  #   dynamodb_table = "terraform-locks"
  #   encrypt        = true
  # }
}
