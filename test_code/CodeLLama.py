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
model_name = "meta-llama/CodeLlama-7b-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, trust_remote_code=True, torch_dtype=torch.bfloat16)

if torch.cuda.is_available():
    model = model.cuda()
    model = DataParallel(model)

input_text = "Write a quick sort algorithm. Provide only the code witout test cases. Name the function quick_sort"

input_ids = tokenizer.encode(input_text, return_tensors="pt")

if torch.cuda.is_available():
    input_ids = input_ids.cuda()

with torch.no_grad():
    if torch.cuda.is_available():
        output = model.module.generate(input_ids, max_new_tokens=1200, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, return_full_text=False)
    else:
        output = model.generate(input_ids, max_new_tokens=1200, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, return_full_text=False)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
