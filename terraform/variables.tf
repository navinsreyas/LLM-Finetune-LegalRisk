variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "ap-southeast-2" # Sydney
}

variable "repository_name" {
  description = "ECR repository name."
  type        = string
  default     = "legalrisk-llm"
}

variable "service_name" {
  description = "App Runner service name."
  type        = string
  default     = "legalrisk-llm"
}

variable "image_tag" {
  description = "Tag of the image in ECR that App Runner should deploy (must already be pushed -- see README.md)."
  type        = string
  default     = "latest"
}

variable "groq_api_key" {
  description = "Groq API key for LLM_PROVIDER=groq inference. No default on purpose -- pass with -var, never commit it to a .tfvars file."
  type        = string
  sensitive   = true
}
