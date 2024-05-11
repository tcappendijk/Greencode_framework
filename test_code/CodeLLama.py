from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import pipeline
import os

# Set environment variable to specify CUDA_VISIBLE_DEVICES
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Specify the GPUs you want to use

custom_cache_dir = "/data/volume_2"
token = "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"
model_name = "meta-llama/CodeLlama-70b-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token)

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, torch_dtype=torch.bfloat16)

code_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt', max_length=1000, pad_token_id=tokenizer.eos_token_id, device_map="balanced", device=0)

# Generate code for an input string
input_string = "Write a python function to calculate the factorial of a number"
generated_code = code_generator(input_string, max_length=100)[0]['generated_text']
print(generated_code)

