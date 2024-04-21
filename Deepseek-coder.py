import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.
bfloat16).cuda()

print(f"Loading model and tokiner took: {time.time() - start_time} seconds")

input_text = "#write a quick sort algorithm"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print(f"Total time taken: {time.time() - start_time} seconds")