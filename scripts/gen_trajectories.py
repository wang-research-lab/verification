import os
from dataclasses import dataclass
from copy import deepcopy
import asyncio
import json
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import re

import simple_parsing
from transformers import AutoTokenizer
from vllm.inputs import TokensPrompt
import openai
import backoff
from tqdm.asyncio import tqdm_asyncio
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

from verification.rollout.preprocess import load_preprocessed_dataset
from verification.rollout.distributed import DistributedVLLM
from verification.rollout.models import DeepSeekProverVLLM
from verification.utils import load_jsonl, save_jsonl, generate_short_id
            
@dataclass
class Args:
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    save_path: str = "data/deepseek-1.5b-verification-training-problems-responses.jsonl"
    num_gpus: int = 1
    tp_size: int = 1
    n_rollouts: int = 1
    dataset_name: str = "verification-training-problems"
    dataset_size: int = -1
    use_api: bool = False
    endpoint: str = "https://api.openai.com/v1"
    api_key: str = os.getenv("OPENAI_API_KEY")
    concurrency_limit: int = 20
    tmp_path: str = "tmp.jsonl"
    
@contextmanager
def ignore_output():
    """
    A context manager that redirects sys.stdout and sys.stderr to os.devnull,
    effectively ignoring any output.
    """
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield
            
def format_and_tokenize(dataset, tokenizer):
    return [
        TokensPrompt(
            prompt_token_ids = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": "<think>\n"},
                ],
                tokenize=True,
                continue_final_message=True,
            )
        ) for item in dataset
    ]
        
def collect_prover_trajectories(engine, args):
    dataset = args.dataset
    
    if args.n_rollouts > 1:
        dataset = [deepcopy(item) for item in dataset for _ in range(args.n_rollouts)]
        
    # for item in dataset:
    #     item["problem_id"] = f"{item['source']}-{item['problem_id']}"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}", ""
    )
    
    prompt_token_ids = format_and_tokenize(dataset, tokenizer)
    
    prover = DeepSeekProverVLLM(vllm_instance=engine)
    outputs = prover(prompt_token_ids, max_tokens=16384, temperature=0.6, top_p=0.95)
    
    for item, output in zip(dataset, outputs):
        item["llm_response"] = "<think>\n" + tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=True)
        item["response_id"] = f"{item['problem_id']}_{generate_short_id()}"
        item["model_name"] = args.model_name
        with ignore_output():
            if re.search(r"\\boxed\{.*?\}", item['gold_standard_solution']) is None:
                gold = parse(f"\\boxed{{{item['gold_standard_solution']}}}", extraction_config=[LatexExtractionConfig(boxed_match_priority=0)])
            else:
                gold = parse(f"{item['gold_standard_solution']}", extraction_config=[LatexExtractionConfig(boxed_match_priority=0)])
            resp = parse(item["llm_response"], extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()])
            item["correct"] = verify(gold, resp)

    return dataset

async def collect_prover_trajectories_api(args):
    dataset = args.dataset
    
    if args.n_rollouts > 1:
        dataset = [deepcopy(item) for item in dataset for _ in range(args.n_rollouts)]
    
    client = openai.AsyncClient(
        base_url=args.endpoint,
        api_key=args.api_key,
    )
    
    semaphore = asyncio.Semaphore(args.concurrency_limit)
    file_lock = asyncio.Lock()
    
    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def process_item(item):
        async with semaphore:
            
            messages = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": "<think>\n"}, # Note: for deepseek
            ]
            
            response = await client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                max_tokens=16384,
                temperature=0.6,
                top_p=0.95,
            )
            
            item["llm_response"] =  "<think>\n" + response.choices[0].message.content # Note: for deepseek
            # item["llm_response"] = response.choices[0].message.content
            item["response_id"] = f"{item['problem_id']}_{generate_short_id()}"
            item["model_name"] = args.model_name
            with ignore_output():
                gold = parse(f"\\boxed{{{item['gold_standard_solution']}}}", extraction_config=[LatexExtractionConfig(boxed_match_priority=0)])
                resp = parse(item["llm_response"], extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()])
                item["correct"] = verify(gold, resp)
            
            return item
        
    tasks = [asyncio.create_task(process_item(item)) for item in dataset]
    
    for future in tqdm_asyncio.as_completed(tasks):
        item = await future
        async with file_lock:
            with open(args.tmp_path, 'a') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
    return dataset

def merge(results, n_rollouts):
    if n_rollouts == 1:
        return results
    
    merged_results = []
    for i in range(0, len(results), n_rollouts):
        problem_id = results[i]["problem_id"]
        source = results[i]["source"]
        prompt = results[i]["prompt"]
        gold_standard_solution = results[i]["gold_standard_solution"]
        llm_responses = []
        
        for j in range(n_rollouts):
            item = results[i + j]
            llm_responses.append({
                "response_id": item["response_id"],
                "model": item["model_name"],
                "response": item["llm_response"],
                "correct": item["correct"]
            })
            
        merged_results.append({
            "problem_id": problem_id,
            "source": source,
            "prompt": prompt,
            "gold_standard_solution": gold_standard_solution,
            "llm_responses": llm_responses
        })
        
    return merged_results


if __name__ == "__main__":    
    args = simple_parsing.parse(Args)
    
    if args.dataset_name.endswith(".jsonl"):
        args.dataset = load_jsonl(args.dataset_name)
    else:
        dataset = load_preprocessed_dataset(args.dataset_name)
        if args.dataset_size > 0:
            dataset = dataset.select(range(args.dataset_size))
        args.dataset = list(dataset)
    
    if args.use_api:
        results = asyncio.run(collect_prover_trajectories_api(args))
        results = merge(results, args.n_rollouts)
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        save_jsonl(args.save_path, results)
        
    else:
        engine = DistributedVLLM(
            num_workers=args.num_gpus//args.tp_size, 
            model=args.model_name, 
            skip_tokenizer_init=True, 
            dtype="bfloat16", 
            tensor_parallel_size=args.tp_size, 
            max_model_len=20_000,
            gpu_memory_utilization=0.90,
        )
        
        results = collect_prover_trajectories(engine, args)
        results = merge(results, args.n_rollouts)
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        save_jsonl(args.save_path, results)
        
        engine.shutdown()