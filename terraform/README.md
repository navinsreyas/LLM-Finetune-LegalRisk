# Terraform: AWS App Runner (ECR image source)

Deploys this repo's actual Dockerfile to AWS App Runner in `ap-southeast-2`
(Sydney) by building the image locally, pushing it to ECR, and pointing App
Runner at that ECR image -- not GitHub source. Runs in Groq mode
(`LLM_PROVIDER=groq`, no local model weights).

## One sentence each

- **Provider** (`hashicorp/aws`): the plugin that translates this HCL into AWS API calls, authenticated using whatever AWS credentials are active in your shell.
- **Resource** (`aws_ecr_repository`, `aws_iam_role`, `aws_apprunner_service`): a declaration of one real AWS object Terraform should create/update/destroy -- here, the image registry, the IAM role App Runner uses to read from it, and the App Runner service itself.
- **State file** (`terraform.tfstate`): Terraform's local record of what it last created, used to diff against on the next `plan`/`apply` -- it will contain `GROQ_API_KEY` in **plain text**, so it's gitignored and must never be committed or shared as-is.

## Why the ECR repo must exist and hold an image before the App Runner service is created

`aws_apprunner_service` references a specific `ecr_repository_url:image_tag`
and, unlike GitHub source, App Runner does not build anything itself here --
on creation it immediately tries to pull and run that exact image, so if the
tag isn't in ECR yet, service creation fails outright. That's why this is a
two-phase apply: create the empty repository first, push an image into it,
then create the service.

## Prerequisites

- Terraform >= 1.5.0, Docker, AWS CLI credentials for an account with ECR + App Runner + IAM permissions.
- A Groq API key.

## Exact command sequence

```bash
cd terraform

# 1. Initialize (downloads the AWS provider)
terraform init

# 2. Create ONLY the ECR repository first -- nothing to push to otherwise.
#    (groq_api_key isn't used by this resource, but Terraform still needs it
#    to evaluate the full config; any placeholder string works at this step.)
terraform apply -target=aws_ecr_repository.app -var="groq_api_key=placeholder"

# 3. Get the repository URL
terraform output ecr_repository_url
# -> e.g. 123456789012.dkr.ecr.ap-southeast-2.amazonaws.com/legalrisk-llm
```

```bash
# 4. Build, tag, and push the real Dockerfile to that repository.
#    Run these from the repo root (one level up), not from terraform/.
cd ..

ECR_URL=$(cd terraform && terraform output -raw ecr_repository_url)

# Authenticate Docker to ECR
aws ecr get-login-password --region ap-southeast-2 \
  | docker login --username AWS --password-stdin "$ECR_URL"

# Build using the repo's actual Dockerfile
docker build -t legalrisk-llm:latest .

# Tag for ECR and push
docker tag legalrisk-llm:latest "$ECR_URL:latest"
docker push "$ECR_URL:latest"
```

```bash
# 5. Now that the image exists in ECR, create the App Runner service (and
#    the IAM access role) with the real Groq key.
cd terraform
terraform apply -var="groq_api_key=YOUR_GROQ_KEY_HERE"
```

```bash
# 6. Get the live URL and test it.
terraform output service_url

curl "$(terraform output -raw service_url)/health"
# -> {"status":"ok"}
```

## Redeploying after a code change

There's no auto-deploy. Repeat step 4 (build/tag/push, same or new tag), then
either re-run `terraform apply` (if the tag changed) or trigger "Deploy" in
the App Runner console (if you pushed to the same tag, since App Runner
doesn't watch ECR for new pushes without an explicit redeploy).

## Teardown

```bash
terraform destroy -var="groq_api_key=YOUR_GROQ_KEY_HERE"
```

This also deletes the ECR repository. If it still contains images, add
`force_delete = true` to `aws_ecr_repository.app` first (not set by default,
so an accidental `destroy` doesn't silently delete a repo full of images) --
or delete the images manually before destroying.
