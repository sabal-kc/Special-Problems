# EC2 Sentiment Benchmark Infrastructure

This Terraform configuration provisions one EC2 instance for the sentiment benchmark service. The EC2 instance runs the service; benchmark scripts are run locally against the generated endpoint.

## What It Creates

- EC2 instance
- Security group for SSH and the sentiment API
- Service bootstrap using EC2 user data
- Outputs for health check, benchmark endpoint, and SSH

## Service Modes

| `service_type` | Runtime | Setup script |
|---|---|---|
| `traditional` | NLTK VADER | `services/setup_ec2_service.sh` |
| `ollama` | Ollama / Llama 3.2 | `services/setup_ec2_ollama_service.sh` |

Both services expose:

```text
GET  /health
POST /v1/sentiment
```

## Configuration

Copy the example variables file:

```bash
cp terraform.tfvars.example terraform.tfvars
```

Set the main values:

```hcl
service_type  = "traditional"
instance_type = "t2.nano"

repo_url = "https://github.com/<owner>/<repo>.git"
repo_ref = "master"

key_name             = "your-key-pair-name"
ssh_private_key_path = "~/.ssh/your-key.pem"
```

Leave these blank to restrict access to the current public IP:

```hcl
allowed_app_cidr = ""
allowed_ssh_cidr = ""
```

## Provision

```bash
terraform init -upgrade
terraform fmt
terraform validate
terraform plan
terraform apply
```

Useful outputs:

```bash
terraform output health_url
terraform output endpoint_url
terraform output ssh_command
```

Verify the service:

```bash
curl "$(terraform output -raw health_url)"
```

## Benchmark Handoff

Use `endpoint_url` as the benchmark service endpoint:

```yaml
traditional:
  endpoint: http://EC2_PUBLIC_IP:8000/v1/sentiment
```

The field is still named `traditional.endpoint` even when testing the Ollama service because both services use the same API.

Run benchmarks locally from the project root:

```bash
python3 benchmarks/task1_sequential_run.py --config benchmarks/task1_config.yaml
python3 benchmarks/task1_concurrent_run.py --config benchmarks/task1_config.yaml
```

## Example Runs

```bash
terraform apply -var-file=examples/traditional-t2-nano.tfvars.example
terraform apply -var-file=examples/traditional-t2-micro.tfvars.example
terraform apply -var-file=examples/ollama-c7a-large.tfvars.example
terraform apply -var-file=examples/ollama-c7a-xlarge.tfvars.example
terraform apply -var-file=examples/ollama-c7a-2xlarge.tfvars.example
```

## Cleanup

```bash
terraform destroy
```

Use the same `-var-file` during destroy if one was used during apply.
