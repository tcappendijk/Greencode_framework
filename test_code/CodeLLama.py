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

# Move model to device (GPU)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, torch_dtype=torch.bfloat16, device_map="balanced")
model.to("cuda")

# Define a function to generate code
def generate_code(input_string, max_length=1000):
    input_ids = tokenizer.encode(input_string, return_tensors="pt").to("cuda")
    output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_code

# Generate code for an input string
input_string = "Write a python function to calculate the factorial of a number"
generated_code = generate_code(input_string, max_length=100)
print(generated_code)


# # Create a pipeline
# code_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt', max_length=1000, pad_token_id=tokenizer.eos_token_id)

# # Generate code for an input string
# input_string = "Write a python function to calculate the factorial of a number"
# generated_code = code_generator(input_string, max_length=100)[0]['generated_text']
# print(generated_code)

