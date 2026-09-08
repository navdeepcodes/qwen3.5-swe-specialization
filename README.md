# Qwen3.5-9B Software-Engineering Specialization

LoRA fine-tuning experiment: specializing a local MLX 4-bit Qwen3.5-9B checkpoint
for general software-engineering ability (generation, debugging, refactoring,
security) on an Apple M4 / 16GB unified memory machine, using `mlx-lm`.

**Status: dataset built and calibrated, training NOT yet launched — awaiting final
approval per the design-review gate below.**

The Ollama `qwen3.5:9b` install is a separate, untouched artifact. This project
only touches a separately downloaded MLX 4-bit checkpoint (`model/`, gitignored —
see Setup).

## Hardware / environment

- Apple M4, 16GB unified memory
- `venv/` (gitignored) — Python 3.12, `mlx==0.32.2`, `mlx-lm==0.31.3`
- Base checkpoint: `mlx-community/Qwen3.5-9B-MLX-4bit` (VLM checkpoint used
  text-only), architecture confirmed via `model/config.json`:
  - 32 decoder layers: 24 `linear_attention` (GatedDeltaNet) + 8 `full_attention`,
    full-attention every 4th layer
  - hidden_size 4096, intermediate_size 12288, 4-bit / group_size 64 quantization
- macOS GPU "wired memory" ceiling is the real constraint on this hardware, not
  raw 16GB RAM — see Memory calibration below.

## Dataset pipeline (`pipeline/build_dataset.py`)

Four permissively-licensed sources, each targeting a distinct competency:

| Source | Competency | License | Selection |
|---|---|---|---|
| `nvidia/OpenCodeInstruct` | code generation | cc-by-4.0 | filtered to `average_test_score == 1.0` (unit-test verified) |
| `m-a-p/CodeFeedback-Filtered-Instruction` | debugging / Q&A | apache-2.0 | multi-language (python/js/sql/cpp/java/php/...) |
| `bigcode/commitpackft` | refactoring / diff-based editing | mit | 15 languages, commit-message quality filter |
| `CyberNative/Code_Vulnerability_Security_DPO` | security | apache-2.0 | `chosen` side used as SFT target (full set, ~4.6k) |

Pipeline stages: normalize to unified `{"messages": [...]}` chat schema → per-source
quality filtering → exact hash dedup → MinHash/LSH near-dedup (32 perm, 8 band,
jaccard ≥ 0.85, self-contained, no extra deps) → cross-source dedup → **held-out
eval carved out first, stratified by source** → train/valid split → leakage
assertion (zero hash overlap between eval and train/valid).

Run it with:
```
venv/bin/python3 pipeline/build_dataset.py
```
Requires `pipeline/raw/` populated first (gitignored, ~1.4GB of source dumps —
see script docstring for the exact download commands used).

### Current curated dataset (`data_curated/`, see `manifest.json`)

- train: 13,806 examples / ~5.70M tokens
- valid: 600 examples
- eval: 600 examples (held out, verified zero overlap with train/valid)
- token length: mean 413, median 361, p95 815, max 4017 (train)

## Memory calibration (`calibration/`)

Short (20-iter) trial runs at target LoRA settings, measuring `mx.get_peak_memory()`
directly on this machine rather than extrapolating from the earlier micro smoke
test. Findings:

1. **Ollama contention**: `ollama ps` showed `qwen3.5:9b` resident in memory
   (~7.4GB RSS) with a keep-alive timer that kept resetting. Unloaded via
   `ollama stop qwen3.5:9b` (does not touch the model files — Ollama reloads on
   next use).
2. **macOS GPU wired-memory ceiling**, not raw RAM, is the binding constraint.
   Peak-before-crash stayed ~12.6GB across multiple configs even as num_layers/
   seq_len shrank, which is the signature of a fixed OS ceiling
   (`sysctl iogpu.wired_limit_mb`) rather than a workload-proportional limit.
   Raised via `sudo sysctl iogpu.wired_limit_mb=13312` (reversible, resets on
   reboot).

## Next steps

1. Re-run calibration at the raised wired-memory limit to get final real
   peak-memory / it/s numbers.
2. Present final LoRA config, memory/runtime estimate, checkpoint strategy,
   and evaluation methodology for approval.
3. Only after explicit approval: launch the real training run as a detached,
   logged process with a status file, periodic checkpoints, and graceful
   failure handling.
4. Evaluate fine-tuned adapter vs. untouched base model on the held-out
   `data_curated/eval.jsonl` set against pre-defined success criteria.

## Repo layout

```
pipeline/build_dataset.py   dataset acquisition + cleaning + dedup + split
data_curated/                train.jsonl / valid.jsonl / eval.jsonl / manifest.json
calibration/adapter_config.json   mlx-lm config used for memory/throughput calibration
```

`model/`, `venv/`, and `pipeline/raw/` are intentionally gitignored (large,
reproducible/re-downloadable, not source code).
