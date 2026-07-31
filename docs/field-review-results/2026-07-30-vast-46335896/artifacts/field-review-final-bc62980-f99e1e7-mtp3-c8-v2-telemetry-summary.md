# Final scorecard GPU telemetry

The sampler called `nvidia-smi` every two seconds. Each call emits GPUs 0–3
in order with slightly different millisecond timestamps, so synchronized
samples below are consecutive validated groups of four rows, not exact string
matches on the per-GPU timestamp. The GPQA CSV's known joined header/first-row
delimiter was repaired in memory only; the archived source remains unchanged.

## GPQA Diamond, C4, 32K completion ceiling

Source: `field-review-final-bc62980-f99e1e7-mtp3-c8-v2-gpqa50-c4-32k-gpu.csv`

- 8,228 accepted GPU rows / 2,057 complete four-GPU samples.
- GPU memory stayed between 92,167 and 92,211 MiB on every rank.
- Aggregate power: 756.56 W mean, 755.24 W p50, 775.31 W p95, 957.06 W max.

| GPU | power mean W | p50 W | p95 W | max W | utilization mean | max |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 194.75 | 194.34 | 201.59 | 253.29 | 99.94% | 100% |
| 1 | 184.05 | 183.73 | 189.84 | 235.68 | 99.90% | 100% |
| 2 | 186.83 | 186.34 | 193.26 | 243.47 | 99.87% | 100% |
| 3 | 190.92 | 190.51 | 197.01 | 239.18 | 99.90% | 100% |

## GSM8K, C8, 2K completion ceiling

Source: `field-review-final-bc62980-f99e1e7-mtp3-c8-v2-gsm8k100-c8-2k-gpu.csv`

- 240 accepted GPU rows / 60 complete four-GPU samples.
- GPU memory stayed between 92,211 and 92,223 MiB on every rank.
- Aggregate power: 1,205.16 W mean, 1,282.36 W p50, 1,340.30 W p95,
  1,422.71 W max.

| GPU | power mean W | p50 W | p95 W | max W | utilization mean | max |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 316.60 | 336.16 | 355.74 | 372.79 | 97.52% | 100% |
| 1 | 290.41 | 308.67 | 327.14 | 344.48 | 97.72% | 100% |
| 2 | 293.84 | 311.31 | 327.84 | 355.53 | 97.62% | 100% |
| 3 | 304.31 | 322.96 | 340.87 | 370.51 | 97.55% | 100% |

These are host draw observations, not a configured power cap. The Vast host
allowed materially more than AIBeast's 280 W/card maintenance limit during
the short, high-concurrency GSM8K workload; long C4 decode averaged only about
184–195 W/card despite 100% reported GPU utilization.
