from dataclasses import dataclass
from typing import List
import os
import ray
import torch
from simple_parsing import parse
from tqdm import tqdm
from transformers import AutoTokenizer
from verification.utils import load_jsonl, save_jsonl
from verification.training.qwen_utils.modeling_qwen2_rm import Qwen2ForRewardModel
from verification.rollout.preprocess import load_preprocessed_dataset

@dataclass
class Args:
    model_name: str = "WangResearchLab/verification-1.5b"
    save_path: str = "evals/verification-1.5b/validation.jsonl"
    dataset_name: str = "verification-evaluation-data"
    dataset_split: str = "validation"
    num_gpus: int = 1
    ignore_thinking: bool = True

# ---------- Ray actor --------------------------------------------------
@ray.remote(num_gpus=1)
class VerifierActor:
    def __init__(self, model_name, ignore_thinking):
        device   = torch.device("cuda")
        self.model = Qwen2ForRewardModel.from_pretrained(
            model_name, _attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", truncation_side="left")
        self.tokenizer.chat_template = self.tokenizer.chat_template.replace(
            "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}", "")
        self.ignore_thinking = ignore_thinking
        self.device = device

    @torch.no_grad()
    def score(self, prompt: str, response: str) -> List[float]:
        inputs = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response.split("</think>")[-1] if self.ignore_thinking else response}
            ],
            tokenize=True, 
            continue_final_message=False,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        
        hidden_states = self.model.model(
            **inputs, 
            return_dict=True
        ).last_hidden_state 
        
        pooled = hidden_states[:, -1, :]
        
        logit = self.model.score(pooled)
        score = torch.sigmoid(logit).item()

        return [score]
    
# ---------- main evaluation -------------------------------------------
def eval_dataset(args: Args):

    if args.dataset_name.endswith(".jsonl"):
        data = load_jsonl(args.dataset_name)
    else:
        data = load_preprocessed_dataset(args.dataset_name, args.dataset_split)
        data = list(data)

    ray.init()
    actors = [VerifierActor.remote(args.model_name, args.ignore_thinking) for _ in range(args.num_gpus)]

    futures, idx = [], 0
    for item in data:
        for resp in item["llm_responses"]:
            actor = actors[idx % len(actors)]
            fut = actor.score.remote(item["prompt"], resp["response"])
            futures.append((fut, resp))
            idx += 1

    # gather results
    for fut, resp in tqdm(futures, desc="verifying"):
        resp["score"] = ray.get(fut)  # list[float]

    ray.shutdown()
    return data

if __name__ == "__main__":
    cfg: Args = parse(Args)
    scored = eval_dataset(cfg)
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    save_jsonl(cfg.save_path, scored)