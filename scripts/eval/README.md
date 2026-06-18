# Evaluation

This directory holds evaluation scripts for the hallucination benchmarks reported in the paper.
Each benchmark uses its official protocol; clone the corresponding repo and follow the steps below.

> **TODO (camera-ready):** drop the per-benchmark inference + scoring scripts here.
> Placeholders are provided so the layout matches the released results table.

| Benchmark | Source | Metric | Script |
|---|---|---|---|
| POPE | https://github.com/RUCAIBox/POPE | Accuracy / F1 | `eval_pope.sh` |
| MME | https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation | Perception / Cognition score | `eval_mme.sh` |
| AMBER | https://github.com/junyangwang0410/AMBER | CHAIR / Cover / Hal / Cog | `eval_amber.sh` |
| Object HalBench | https://github.com/RLHF-V/RLHF-V/tree/main | CHAIRs / CHAIRi | `eval_objhal.sh` |
| MMHal-Bench | https://huggingface.co/datasets/Shengcao1006/MMHal-Bench | Score / Hal rate (GPT-4 judge) | `eval_mmhal.sh` |

## Usage

```bash
# 1. Point MODEL_PATH at your TARS-trained checkpoint
export MODEL_PATH=/path/to/tars-llava-checkpoint

# 2. Run a benchmark
bash scripts/eval/eval_pope.sh
```

Set the shared variables (`MODEL_PATH`, `DATA_ROOT`, `OUTPUT_DIR`) once at the top of each script.
