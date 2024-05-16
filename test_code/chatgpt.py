from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

custom_cache_dir = "/data/volume_2"
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large", custom_cache_dir=custom_cache_dir)
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large", custom_cache_dir=custom_cache_dir)


generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
set_seed(42)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
