
from vllm import SamplingParams

class vLLMModel:
    def __init__(self, vllm_instance):
        self.vllm = vllm_instance
    
    def __call__(self, **generation_kwargs):
        pass  
    
class DeepSeekProverVLLM(vLLMModel):
    def __call__(self, prompt_token_ids, **generation_kwargs):
        return self.vllm.generate(prompt_token_ids, SamplingParams(**generation_kwargs))