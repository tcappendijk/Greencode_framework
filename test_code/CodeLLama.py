from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.parallel import DataParallel

# num_devices = torch.cuda.device_count()

# device_ids = []
# for i in range(num_devices):
#     device_ids.append(i)
#     torch.cuda.set_device(i)


custom_cache_dir = "/data/volume_2"

token = "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"
model_name = "meta-llama/CodeLlama-70b-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, trust_remote_code=True, torch_dtype=torch.bfloat16)

if torch.cuda.is_available():
    model = model.cuda()
    model = DataParallel(model)

input_text = "Write a quick sort algorithm without test cases. Name the function quick_sort"

input_ids = tokenizer.encode(input_text, return_tensors="pt")

if torch.cuda.is_available():
    input_ids = input_ids.cuda()

print(input_ids)

with torch.no_grad():
    if torch.cuda.is_available():
        output = model.module.generate(input_ids, max_new_tokens=1200, pad_token_id=tokenizer.eos_token_id)
    else:
        output = model.generate(input_ids, max_new_tokens=1200, pad_token_id=tokenizer.eos_token_id)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
