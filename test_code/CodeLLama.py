from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.parallel import DataParallel

custom_cache_dir = "/data/volume_2"

token = "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"
model_name = "meta-llama/CodeLlama-70b-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, trust_remote_code=True, torch_dtype=torch.bfloat16)

device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

if torch.cuda.is_available():
    model = DataParallel(model, device_ids=device_ids)
    model = model.cuda()  # Move model to GPU
    print("Model is now DataParallel")

input_text = "Write a quick sort algorithm without test cases. Name the function quick_sort"

input_ids = tokenizer.encode(input_text, return_tensors="pt")

if torch.cuda.is_available():
    input_ids = input_ids.cuda()

print(input_ids)

with torch.no_grad():
    if torch.cuda.is_available():
        output = model.module.generate(input_ids.cuda(), max_new_tokens=1200, pad_token_id=tokenizer.eos_token_id)
    else:
        output = model.generate(input_ids, max_new_tokens=1200, pad_token_id=tokenizer.eos_token_id)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
