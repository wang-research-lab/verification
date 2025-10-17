import argparse
import time
import torch
import random
import multiprocessing as mp
from multiprocessing import Process, Queue
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
from verification.training.qwen_utils.modeling_qwen2_rm import Qwen2ForRewardModel

def generative_latency_worker(gpu_id: int, model_name: str, input_tokens: int, output_tokens: int, queue: Queue) -> None:
    """Worker function for generative latency measurement on a specific GPU."""
    torch.cuda.set_device(str(gpu_id))
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LLM(model_name, dtype="bfloat16", max_model_len=20000, gpu_memory_utilization=0.95)
    times = []

    prompt = tokenizer.decode(random.choices(range(tokenizer.vocab_size), k=input_tokens))

    for N in [1, 2, 4, 8, 16, 32, 64, 128]:
        sampling_params = SamplingParams(
            max_tokens=output_tokens,
            temperature=0.6,
            top_p=0.95,
            ignore_eos=True,
            n=N,
        )

        start_time = time.time()
        outputs = model.generate(prompt, sampling_params)
        end_time = time.time()
        total_time = end_time - start_time
        times.append(total_time)
        print(f"GPU {gpu_id} - N={N}: {total_time}")

    queue.put(times)

def generative_latency(args):
    """Run generative latency measurement across multiple GPUs in parallel and average results."""
    model_name = args.model_name
    num_gpus = args.num_gpus
    
    # Create queue for collecting results
    queue = Queue()
    processes = []
    
    # Launch processes on each GPU
    for gpu_id in range(num_gpus):
        p = Process(target=generative_latency_worker, 
                   args=(gpu_id, model_name, args.input_tokens, args.output_tokens, queue))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Collect results from all processes
    all_times = []
    while not queue.empty():
        all_times.append(queue.get())
    
    # Average the results across all GPUs
    if all_times:
        avg_times = np.mean(all_times, axis=0)
        print(f"Averaged times across {num_gpus} GPUs: {avg_times}")
        return avg_times
    else:
        print("No results collected!")
        return []

def descriminative_latency_worker(gpu_id: int, model_name: str, input_tokens: int, queue: Queue) -> None:
    """Worker function for discriminative latency measurement on a specific GPU."""
    torch.cuda.set_device(str(gpu_id))
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Qwen2ForRewardModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, _attn_implementation="flash_attention_2").cuda()

    times = []

    with torch.inference_mode():
        for i in tqdm(range(1000)):
            input_ids = torch.tensor([random.choices(range(tokenizer.vocab_size), k=input_tokens)]).cuda()

            start_time = time.time()
            hidden_states = model.model(
                input_ids, 
                return_dict=True
            ).last_hidden_state 
            
            pooled = hidden_states[:, -1, :]
            logit = model.score(pooled)
            end_time = time.time()
            
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        print(f"GPU {gpu_id} - Total model time: {avg_time}")
        
        queue.put(avg_time)

def descriminative_latency(args):
    """Run discriminative latency measurement across multiple GPUs in parallel and average results."""
    model_name = args.model_name
    num_gpus = args.num_gpus
    
    # Create queue for collecting results
    queue = Queue()
    processes = []
    
    # Launch processes on each GPU
    for gpu_id in range(num_gpus):
        p = Process(target=descriminative_latency_worker, 
                   args=(gpu_id, model_name, args.input_tokens, queue))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Collect results from all processes
    all_times = []
    while not queue.empty():
        all_times.append(queue.get())
    
    # Average the results across all GPUs
    if all_times:
        avg_time = np.mean(all_times)
        print(f"Averaged total model time across {num_gpus} GPUs: {avg_time}")
        return avg_time
    else:
        print("No results collected!")
        return None


if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_tokens", type=int, required=True)
    parser.add_argument("--output_tokens", type=int, required=True)
    parser.add_argument("--num_gpus", type=int, default=8)
    args = parser.parse_args()

    descriminative_latency(args)
