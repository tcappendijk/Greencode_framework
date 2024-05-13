from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import pipeline

custom_cache_dir = "/data/volume_2"
token = "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"
model_name = "meta-llama/CodeLlama-70b-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token)

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, torch_dtype=torch.bfloat16, device_map="balanced")

code_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt', pad_token_id=tokenizer.eos_token_id)

# Generate code for an input string
input_string = """
Given a list of numbers arranged in a specific order, return the list with its numbers sorted from smallest to largest.

Example 1:
Input: [4,2,1,3]
Output: [1,2,3,4]

Example 2:
Input: [-1,5,3,4,0]
Output: [-1,0,3,4,5]

Example 3:
Input: []
Output: []

Constraints:
- The list can have at most 50,000 numbers.
- Each number in the list is between -100,000 and 100,000.

Follow up:
Can you sort the list efficiently in terms of both time and memory usage?
"""

generated_code = code_generator(input_string, max_length=1000)[0]['generated_text']
print(generated_code)

