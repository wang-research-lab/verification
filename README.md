# Budget-Aware Test-Time Scaling via Discriminative Verification

This repository contains the implementation for the paper ["Budget-Aware Test-Time Scaling via Discriminative Verification"](https://arxiv.org/pdf/2510.14913).

<p align="center">
  📃 <a href="https://arxiv.org/abs/2510.14913" target="_blank">[Paper]</a> • 💻 <a href="https://github.com/wang-research-lab/verification" target="_blank">[GitHub]</a> • 🤗 <a href="https://huggingface.co/collections/WangResearchLab/verification-68f1abb00d7f3f7ef6e83ff3" target="_blank">[Hugging Face]</a>
</p>

## Installation

```bash
git clone https://github.com/wang-research-lab/verification.git
cd verification

conda create -n verification python=3.10
conda activate verification

pip install -e . # will install `verification` and various dependencies
```

## Usage

### 1. Generate Candidate Solutions

Use `gen_trajectories.py` to generate candidate solutions via vLLM. To sample a solution from "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" for every training problem:

```bash
python scripts/gen_trajectories.py \
    --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --save_path "data/deepseek-r1-1.5b-verification-training-problems-responses.jsonl" \
    --num_gpus 8 \
    --dataset_name "verification-training-problems"
```

`gen_trajectories.py` can also generate candidate solutions for evaluation datasets (`aime2024`, `aime2025`, `livebench-math`, and `gpqa`):

```bash
python scripts/gen_trajectories.py \
    --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
    --save_path "data/deepseek-32b-aime2024-responses.jsonl" \
    --num_gpus 8 \
    --tp_size 8 \
    --n_rollouts 128 \
    --dataset_name "aime2024"
```

Alternatively, you can use `gen_trajectories.py` with an OpenAI-compatible API instead of vLLM:

```bash
python scripts/gen_trajectories.py \
    --model_name "deepseek-ai/DeepSeek-R1" \
    --save_path "data/deepseek-r1-verification-training-problems-responses.jsonl" \
    --dataset_name "verification-training-problems" \
    --use_api True \
    --endpoint "https://api.together.xyz/v1" \
    --api_key "Your-Together-API-Key" \
    --concurrency_limit 20
```


### 2. Train Discriminative Verifier

Train a 1.5B parameter discriminative verifier using accelerate with FSDP:

```bash
accelerate launch --config_file configs/fsdp_8gpu.yaml scripts/train_ranking.py \
    --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --dataset_name "verification-training-data" \
    --ckpt_path "outputs/verification-1.5b" \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr 5e-5
```

### 3. Run Verification on Evaluation Dataset

Use `run_judge_hf.py` to score candidate solutions with the trained verifier:

```bash
python scripts/run_judge_hf.py \
    --model_name "WangResearchLab/verification-1.5b" \
    --dataset_name "verification-evaluation-data" \
    --dataset_split "validation" \
    --save_path "evals/verification-1.5b/validation-eval.jsonl" \
    --num_gpus 8
```

## Citation

```bibtex
@article{montgomery2025budget,
  title={Budget-Aware Test-Time Scaling via Discriminative Verification},
  author={Montgomery, Kyle and Tan, Sijun and Chen, Yuqi and Zhuang, Siyuan and Zhang, Tianjun and Popa, Raluca Ada and Wang, Chenguang},
  journal={arXiv preprint arXiv:2510.14913},
  year={2025}
}
```
