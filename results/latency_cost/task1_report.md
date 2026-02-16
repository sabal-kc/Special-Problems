# Task 1 E2E Benchmark Report

- Run ID: `320ad169-fe80-46e7-9d01-1632cd224c94`
- Generated (UTC): `2026-02-16T03:58:31+00:00`
- Measurement mode: End-to-end from local client
- Workload: fixed batch of 1000 sentiment samples

## Run Configuration

- Traditional endpoint: `http://54.242.38.69:8000/v1/sentiment`
- Gemini model: `gemini-3-pro-preview`
- Warm runs per system: `10`
- Timeout per call: `30s`

## Pricing Snapshot

- Snapshot date: `2026-02-14`
- Source note: Manual benchmark config. Replace Gemini and EC2 prices with your current rates before final reporting.
- Traditional EC2: `t3.micro` (us-east-1) at `$0.0104/hour`
- Gemini token rates: input `$1.25e-06` / token, output `$5e-06` / token

## Cold Latency (1 run)

| System | Success Rate | E2E ms |
|---|---:|---:|
| Traditional | 100.00% | 365.5090 |
| Gemini | 0.00% | - |

## Warm Latency and Throughput (10 runs)

| System | Success Rate | p50 ms | p95 ms | Mean samples/sec | Traditional server ms (mean) |
|---|---:|---:|---:|---:|---:|
| Traditional | 100.00% | 355.9460 | 400.2719 | 2851.8961 | 101.1196 |
| Gemini | 0.00% | - | - | - | - |

## Cost Per Request (warm mean)

| System | API USD | Infra USD | Total USD |
|---|---:|---:|---:|
| Traditional | 0.00000000 | 0.00000102 | 0.00000102 |
| Gemini | - | - | - |

## Decision Table (hourly cost interpretation)

| Traffic | Requests/hour | Traditional USD/hour | Gemini USD/hour | Cheaper |
|---|---:|---:|---:|---|
| low | 100 | 0.010400 | - | n/a |
| medium | 1000 | 0.010400 | - | n/a |
| high | 10000 | 0.010400 | - | n/a |

## Caveats

- This benchmark is local-client end-to-end and includes network effects.
- Traditional and Gemini quality parity is not evaluated in this report.
- Cost is sensitive to token usage and configured pricing values.
