# ============================================================================
# AWS App Runner, deployed from an ECR image (not GitHub source).
#
# Why: App Runner's GitHub-source mode builds via its own buildpack, not your
# actual Dockerfile, and its source checkout isn't guaranteed to run
# `git lfs pull` -- risking a broken (pointer-stub) copy of the Git-LFS-tracked
# data/rag/chroma_db index. Building the real Dockerfile locally and pushing
# the resulting image to ECR sidesteps both problems: App Runner just pulls
# and runs the exact image you built and tested.
# ============================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# --- ECR: where the locally-built Dockerfile image gets pushed -------------

resource "aws_ecr_repository" "app" {
  name                 = var.repository_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project = "LegalRisk-LLM"
  }
}

# --- IAM: the role App Runner assumes to pull from this private ECR repo ---
#
# App Runner does not read ECR images as "you" (the Terraform/AWS CLI
# caller) -- the App Runner *service* itself needs its own IAM identity and
# explicit permission to call ecr:GetDownloadUrlForLayer / BatchGetImage /
# BatchCheckLayerAvailability / GetAuthorizationToken against your account,
# because the pull happens inside AWS's App Runner build/deploy
# infrastructure, not on your machine. This role + trust policy is what
# grants that: the trust policy lets the App Runner build service
# (build.apprunner.amazonaws.com) assume the role, and the attached AWS
# managed policy grants exactly the ECR read permissions needed. Without it,
# service creation fails with an image-pull authorization error even though
# the image genuinely exists in ECR.

resource "aws_iam_role" "apprunner_ecr_access" {
  name = "${var.service_name}-apprunner-ecr-access"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "build.apprunner.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Project = "LegalRisk-LLM"
  }
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr_access" {
  role       = aws_iam_role.apprunner_ecr_access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

# --- App Runner service ------------------------------------------------------

resource "aws_apprunner_service" "app" {
  service_name = var.service_name

  source_configuration {
    # OFF per request: nothing rebuilds/redeploys automatically. A new deploy
    # means: build -> push a new tag (or the same tag) to ECR -> re-run
    # `terraform apply` (or trigger a deploy in the console) to point App
    # Runner at it.
    auto_deployments_enabled = false

    authentication_configuration {
      # Required for private ECR (unlike ECR Public, which needs no role) --
      # this is the role defined above.
      access_role_arn = aws_iam_role.apprunner_ecr_access.arn
    }

    image_repository {
      image_repository_type = "ECR"
      image_identifier      = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"

      image_configuration {
        port = "8080"

        runtime_environment_variables = {
          # Forces the Groq code path in src/rag/rag_pipeline.py
          # (self.provider = os.environ.get("LLM_PROVIDER", "local")) so the
          # local Llama model/tokenizer is never loaded -- matches the
          # Dockerfile's own LLM_PROVIDER=groq.
          LLM_PROVIDER = "groq"
          RAG_DEVICE   = "cpu"
          PORT         = "8080"

          # NOTE: same caveat as before -- this lands in the Terraform state
          # file in plain text (see README's "state file" note). App
          # Runner's `runtime_environment_secrets` field would avoid that,
          # but only accepts an ARN of an existing Secrets Manager/SSM
          # secret, not a raw value, so it needs a separate secret resource
          # to use properly (out of scope here; worth doing for production).
          GROQ_API_KEY = var.groq_api_key
        }
      }
    }
  }

  instance_configuration {
    # 1 vCPU / 2 GB -- unchanged from the GitHub-source config. Even in Groq
    # mode, RAGPipeline.__init__ always loads sentence-transformers + CPU
    # torch + the ChromaDB index for retrieval (only the LLM call itself
    # goes to Groq); that footprint doesn't reliably fit the 0.25 vCPU /
    # 0.5 GB tier.
    cpu    = "1 vCPU"
    memory = "2 GB"
  }

  health_check_configuration {
    # /health (app/main.py: GET /health -> {"status": "ok"}), not /metrics --
    # /metrics is a Prometheus scrape target, not a pass/fail liveness probe.
    protocol            = "HTTP"
    path                = "/health"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }

  tags = {
    Project = "LegalRisk-LLM"
  }

  # The image tag referenced above must already exist in ECR before this
  # resource is created -- App Runner tries to pull and deploy it
  # immediately, it doesn't wait or build anything itself. See README.md for
  # the required apply order.
  depends_on = [
    aws_ecr_repository.app,
    aws_iam_role_policy_attachment.apprunner_ecr_access,
  ]
}
