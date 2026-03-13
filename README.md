# NLP Deployment Benchmarks

This repository compares different ways to run sentiment analysis:

- a traditional NLP baseline using NLTK VADER
- a self-hosted LLM service using Ollama
- external LLM results used for comparison

The main focus is benchmarking latency, throughput, cost, and output quality across local code, EC2-hosted services, and LLM-based approaches.

## Project Structure

```text
benchmarks/
  Scripts and config for latency, throughput, and cost benchmarks.

services/
  HTTP sentiment services and EC2 setup scripts.

terraform/
  Terraform configuration for provisioning EC2 benchmark instances.

results/
  Saved benchmark outputs, summaries, and reports.

llm_input/
  Input CSV files used for LLM and service benchmarks.

llm_output/
  Saved LLM outputs.

llama_output/
  Saved self-hosted LLM outputs.

notebooks/
  Analysis notebooks.

custom_datasets.py
  Dataset loading and sampling helpers.

traditional_nlp.py
  Traditional NLP baseline runner for the broader task comparison.

task_overview.md
  Notes on the benchmark tasks and datasets.
```

## Basic Flow

```text
1. Prepare benchmark input data.
2. Run or deploy a sentiment service.
3. Run benchmark scripts from the local machine.
4. Save and compare the results.
```

For EC2-based runs, use the Terraform setup under `terraform/ec2/`.
