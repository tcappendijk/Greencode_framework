from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

custom_cache_dir = "/data/volume_2"
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large", cache_dir=custom_cache_dir)
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large", cache_dir=custom_cache_dir)


generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
set_seed(42)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)


# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from transformers import pipeline

# custom_cache_dir = "/data/volume_2"
# token = "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"
# model_name = "openai-community/gpt2-large"

# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token)

# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, torch_dtype=torch.bfloat16, device_map="balanced")

# generator = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt', pad_token_id=tokenizer.eos_token_id)
