# Task 1 E2E Benchmark Report

- Run ID: `acd4bb07-8bc3-458c-acfb-bb460ff2a62f`
- Generated (UTC): `2026-02-15T01:21:47+00:00`
- Measurement mode: End-to-end from local client
- Workload: fixed batch of 100 sentiment samples

## Run Configuration

- Traditional endpoint: `http://54.234.160.213:8000/v1/sentiment`
- Gemini model: `gemini-2.5-flash`
- Warm runs per system: `1`
- Timeout per call: `30s`

## Pricing Snapshot

- Snapshot date: `2026-02-14`
- Source note: Manual benchmark config. Replace Gemini and EC2 prices with your current rates before final reporting.
- Traditional EC2: `t3.micro` (us-east-1) at `$0.0104/hour`
- Gemini token rates: input `$1.25e-06` / token, output `$5e-06` / token

## Cold Latency (1 run)

| System | Success Rate | E2E ms |
|---|---:|---:|
| Traditional | 100.00% | 101.5013 |
| Gemini | 100.00% | 13730.2107 |

## Warm Latency and Throughput (1 runs)

| System | Success Rate | p50 ms | p95 ms | Mean samples/sec | Traditional server ms (mean) |
|---|---:|---:|---:|---:|---:|
| Traditional | 100.00% | 98.6257 | 98.6257 | 1013.9344 | 9.4003 |
| Gemini | 100.00% | 18529.5730 | 18529.5730 | 5.3968 | - |

## Cost Per Request (warm mean)

| System | API USD | Infra USD | Total USD |
|---|---:|---:|---:|
| Traditional | 0.00000000 | 0.00000028 | 0.00000028 |
| Gemini | 0.00468250 | 0.00000000 | 0.00468250 |

## Decision Table (hourly cost interpretation)

| Traffic | Requests/hour | Traditional USD/hour | Gemini USD/hour | Cheaper |
|---|---:|---:|---:|---|
| low | 100 | 0.010400 | 0.468250 | Traditional |
| medium | 1000 | 0.010400 | 4.682500 | Traditional |
| high | 10000 | 0.010400 | 46.825000 | Traditional |

## Caveats

- This benchmark is local-client end-to-end and includes network effects.
- Traditional and Gemini quality parity is not evaluated in this report.
- Cost is sensitive to token usage and configured pricing values.
