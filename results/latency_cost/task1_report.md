# Task 1 E2E Benchmark Report

- Run ID: `0993ca48-d8f2-4252-9e95-db7f7b5e7791`
- Generated (UTC): `2026-02-14T23:08:32+00:00`
- Measurement mode: End-to-end from local client
- Workload: fixed batch of 100 sentiment samples

## Run Configuration

- Traditional endpoint: `http://127.0.0.1:8000/v1/sentiment`
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
| Traditional | 100.00% | 15.6064 |
| Gemini | 100.00% | 18675.1282 |

## Warm Latency and Throughput (1 runs)

| System | Success Rate | p50 ms | p95 ms | Mean samples/sec | Traditional server ms (mean) |
|---|---:|---:|---:|---:|---:|
| Traditional | 100.00% | 20.4358 | 20.4358 | 4893.3655 | 15.3143 |
| Gemini | 100.00% | 14001.2028 | 14001.2028 | 7.1422 | - |

## Cost Per Request (warm mean)

| System | API USD | Infra USD | Total USD |
|---|---:|---:|---:|
| Traditional | 0.00000000 | 0.00000006 | 0.00000006 |
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
