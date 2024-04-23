# deepseek-coder-6.7b-base

import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", local_files_only=True, torch_dtype=torch.
bfloat16)
model = model.to(device0)

print(f"Loading model and tokiner took: {time.time() - start_time} seconds")

input_text = "#write a quick sort algorithm"
inputs = tokenizer(input_text, return_tensors="pt").to(device0)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print(f"Total time taken: {time.time() - start_time} seconds")

# deepseek-coder-6.7b-instruct

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to(device1)
messages=[
    { 'role': 'user', 'content': "write a quick sort algorithm in python."}
]

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device1)
# tokenizer.eos_token_id is the id of <|EOT|> token
outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))