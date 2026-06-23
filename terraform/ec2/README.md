# EC2 Sentiment Benchmark Infrastructure

This folder contains the Terraform setup for the EC2 sentiment benchmark service.

The flow is:

```text
Terraform creates EC2 service
  -> local benchmark scripts send requests to EC2
  -> benchmark results are saved locally under results/
```

## What It Sets Up

- One EC2 instance
- Security group for SSH and the sentiment API
- Repository checkout on the EC2 instance
- Service setup through EC2 user data
- Outputs for the health URL, benchmark endpoint, selected AZ, and SSH command

## Service Modes

| `service_type` | Runtime | Setup script |
|---|---|---|
| `traditional` | NLTK VADER | `services/setup_ec2_service.sh` |
| `ollama` | Ollama / Llama 3.2 | `services/setup_ec2_ollama_service.sh` |

Both services expose the same API:

```text
GET  /health
POST /v1/sentiment
```

## Configure

From this folder:

```bash
cp terraform.tfvars.example terraform.tfvars
```

Set the main values in `terraform.tfvars`:

```hcl
service_type      = "traditional"
instance_type     = "t2.nano"
availability_zone = ""

repo_url = "https://github.com/<owner>/<repo>.git"
repo_ref = "master"

key_name             = "your-key-pair-name"
ssh_private_key_path = "~/.ssh/your-key.pem"
```

Leave access CIDRs blank to restrict SSH and API access to the current public IP:

```hcl
allowed_app_cidr = ""
allowed_ssh_cidr = ""
```

Leaving `availability_zone` blank lets Terraform choose a subnet in an AZ where the selected instance type is supported.

## Provision Cloud Instance

```bash
terraform init -upgrade
terraform fmt
terraform validate
terraform plan
terraform apply
```

After apply, get the generated values:

```bash
terraform output health_url
terraform output endpoint_url
terraform output availability_zone
terraform output ssh_command
```

Check the service:

```bash
curl "$(terraform output -raw health_url)"
```

## Run Benchmarks Locally

Copy the endpoint value:

```bash
terraform output -raw endpoint_url
```

Paste it into `benchmarks/task1_config.yaml`:

```yaml
traditional:
  endpoint: http://EC2_PUBLIC_IP:8000/v1/sentiment
```

Use the same field for either service mode; both services expose `/v1/sentiment`.

Then run the benchmark scripts from the project root:

```bash
python3 benchmarks/task1_sequential_run.py --config benchmarks/task1_config.yaml
python3 benchmarks/task1_concurrent_run.py --config benchmarks/task1_config.yaml
```

The benchmark outputs are written under the `results/` path configured in `benchmarks/task1_config.yaml`.

## Example Runs

```bash
terraform apply -var-file=examples/traditional-t2-nano.tfvars.example
terraform apply -var-file=examples/traditional-t2-micro.tfvars.example
terraform apply -var-file=examples/ollama-c7a-large.tfvars.example
terraform apply -var-file=examples/ollama-c7a-xlarge.tfvars.example
terraform apply -var-file=examples/ollama-c7a-2xlarge.tfvars.example
```

Use the same `-var-file` when destroying that instance.

## Debugging

SSH into the instance:

```bash
$(terraform output -raw ssh_command)
```

Check setup logs:

```bash
sudo tail -100 /var/log/sentiment-bootstrap.log
```

Check service status:

```bash
sudo systemctl status traditional-sentiment
sudo systemctl status ollama-sentiment
```

Check local health from inside EC2:

```bash
curl http://127.0.0.1:8000/health
```

## Cleanup

```bash
terraform destroy
```

Use the same `-var-file` during destroy if one was used during apply.
