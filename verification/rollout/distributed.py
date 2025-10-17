import ray
import torch
from vllm import LLM, SamplingParams
import gc

@ray.remote
class RayVLLMWorker:
    def __init__(self, **model_kwargs):
        self.model = LLM(**model_kwargs)

    def generate(self, prompt_token_ids, sampling_params):
        return self.model.generate(prompt_token_ids, sampling_params)
    
    def shutdown(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

class DistributedVLLM:
    def __init__(self, num_workers: int, **model_kwargs):
        
        gpu_per_worker = model_kwargs.get("tensor_parallel_size", 1)
        
        ray.init(num_gpus=gpu_per_worker * num_workers)
        
        self.workers = [
            RayVLLMWorker.options(num_gpus=gpu_per_worker).remote(**model_kwargs) for _ in range(num_workers)
        ]
    
    def generate(self, prompt_token_ids: list[list[int]], sampling_params: SamplingParams):
        # Split the messages into chunks for each worker
        chunk_size = (len(prompt_token_ids) + len(self.workers) - 1) // len(self.workers)
        chunks = [prompt_token_ids[i:i + chunk_size] for i in range(0, len(prompt_token_ids), chunk_size)]
        
        # Dispatch chunks to workers
        futures = [
            worker.generate.remote(chunk, sampling_params)
            for worker, chunk in zip(self.workers, chunks)
        ]
        
        # Gather results
        results = ray.get(futures)
        
        # Combine results
        combined_results = []
        for result in results:
            combined_results.extend(result)
                
        return combined_results
        
    def shutdown(self):
        for worker in self.workers:
            ray.get(worker.shutdown.remote())
        ray.shutdown()