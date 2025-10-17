import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from simple_parsing import parse
from transformers import AutoTokenizer, get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

from verification.utils import load_jsonl
from verification.training.ranking import ListwiseRankingTrainer
from verification.training.qwen_utils.modeling_qwen2_rm import Qwen2ForCausalLM, Qwen2ForRewardModel
from verification.training.qwen_utils.configuration_qwen2_rm import Qwen2RMConfig
from verification.rollout.preprocess import load_preprocessed_dataset

@dataclass
class Args:
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    dataset_name: str = "verification-training-data" 
    ckpt_path: str = "outputs/verification-1.5b"
    per_device_batch_size: int = 1 # global_batch_size = per_device_batch_size x num_gpus x gradient_accumulation_steps
    gradient_accumulation_steps: int = 4
    lr: float = 5e-5
    max_grad_norm: float = 1.0
    save_steps: int = 1000
    max_steps: int = 1000
    seed: int = 42
    lam: float = 0.01
    ignore_thinking: bool = True
    backbone_type: str = "causal_lm"
    freeze_backbone: bool = False

class RankingTrajectoryDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, ignore_thinking):
        if dataset_name.endswith(".jsonl"):
            self.data = load_jsonl(dataset_name)
        else:
            self.data = load_preprocessed_dataset(dataset_name)
            self.data = list(self.data)
        self.tokenizer = tokenizer
        self.ignore_thinking = ignore_thinking

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompts = [self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": response_item["response"].split("</think>")[-1] if self.ignore_thinking else response_item["response"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        ) for response_item in item["llm_responses"]]
        
        inputs = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=18000,
            return_tensors="pt",
            add_special_tokens=False,
        )
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = torch.tensor([x["correct"] for x in item["llm_responses"]], dtype=torch.float)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
def train(args):
    set_seed(args.seed)
    
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with="wandb", step_scheduler_with_optimizer=False)
    
    if args.backbone_type == "causal_lm":
        backbone = Qwen2ForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        config = Qwen2RMConfig(**backbone.config.to_dict())
        config._attn_implementation = "flash_attention_2"
        model = Qwen2ForRewardModel(config)
        model.model.load_state_dict(backbone.model.state_dict(), strict=False)
        del backbone
        
    elif args.backbone_type == "reward_model":
        model = Qwen2ForRewardModel.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("score"):
                param.requires_grad = False
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left", truncation_side="left")
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}", ""
    )
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = accelerator.prepare(model)
    
    dataset = RankingTrajectoryDataset(args.dataset_name, tokenizer, args.ignore_thinking)
    dataloader = DataLoader(dataset, batch_size=args.per_device_batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=20, num_training_steps=min(len(dataloader)/(accelerator.num_processes * args.gradient_accumulation_steps), args.max_steps))
    optimizer, scheduler, dataloader = accelerator.prepare(optimizer, scheduler, dataloader)
    
    trainer = ListwiseRankingTrainer(model, optimizer, accelerator, args)
    model.train()
    accelerator.init_trackers(project_name="verification")
    
    step = 0
    for batch in tqdm(dataloader):
        with accelerator.accumulate(model):
            metrics = trainer.step(batch)
            metrics["lr"] = scheduler.get_last_lr()[0]
            accelerator.log(metrics)

        if accelerator.sync_gradients:
            step += 1
            scheduler.step()
            
            if step % args.save_steps == 0:
                
                save_dir = os.path.join(args.ckpt_path, f"step_{step}")
                os.makedirs(save_dir, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(
                    save_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model)
                )
                
                tokenizer.save_pretrained(save_dir)
            
            if step >= args.max_steps:
                break
            
    save_dir = os.path.join(args.ckpt_path, "final")
    os.makedirs(save_dir, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(
        save_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model)
    )
    
    tokenizer.save_pretrained(save_dir)
    
    accelerator.end_training()

if __name__ == "__main__":
    args = parse(Args)
    train(args)  