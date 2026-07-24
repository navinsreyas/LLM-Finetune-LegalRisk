output "ecr_repository_url" {
  description = "ECR repository URL -- docker tag/push your image here before creating the App Runner service."
  value       = aws_ecr_repository.app.repository_url
}

output "service_url" {
  description = "Public HTTPS URL of the deployed App Runner service."
  value       = "https://${aws_apprunner_service.app.service_url}"
}
