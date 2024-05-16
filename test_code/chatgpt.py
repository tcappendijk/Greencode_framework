from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import pipeline

custom_cache_dir = "/data/volume_2"
token = "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"
model_name = "openai-community/gpt2-xl"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token)

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, device_map="sequential")

code_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt', pad_token_id=tokenizer.eos_token_id)

# Generate code for an input string
input_string = "Write a quicksort algorithm in python."

generated_code = code_generator(input_string)
print(generated_code)
print(generated_code[0]['generated_text'])