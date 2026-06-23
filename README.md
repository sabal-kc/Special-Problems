# NLP Deployment Benchmarks

This repository compares traditional NLP methods and LLM-based approaches across several NLP tasks:

- sentiment analysis
- named entity recognition
- part-of-speech tagging
- topic classification
- language identification

The deployment and cost benchmarking portion focuses on sentiment analysis, comparing a traditional NLTK VADER service, a self-hosted Ollama/Llama service, and external LLM results.

The main measurements are output quality, latency, throughput, and cost across local baselines, EC2-hosted services, and LLM-based approaches.

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
  Traditional NLP baseline runner for the task comparison.

task_overview.md
  Notes on the benchmark tasks and datasets.
```

## Basic Flow

```text
1. Build task datasets and baseline outputs.
2. Compare traditional NLP results with LLM outputs.
3. Deploy a sentiment service for EC2 benchmarking.
4. Run latency and throughput benchmarks locally.
5. Save and compare the results.
```

For EC2-based runs, use the Terraform setup under `terraform/ec2/`.

## Results

- [Comparison of NLP Tasks - Traditional vs LLMs.pdf](<Comparison of NLP Tasks - Traditional vs LLMs.pdf>)
- [Comparison of NLP Tasks - Traditional vs Self Hosted vs LLMs.pdf](<Comparison of NLP Tasks - Traditional vs Self Hosted vs LLMs.pdf>)
