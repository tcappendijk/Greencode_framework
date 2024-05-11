from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.parallel import DataParallel

custom_cache_dir = "/data/volume_2"

token = "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"
model_name = "meta-llama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, trust_remote_code=True, torch_dtype=torch.bfloat16)

if torch.cuda.is_available():
    model = model.cuda()

    model = DataParallel(model)

input_text = "#write a quick sort algorithm. Provide only the code. Name the function quick_sort"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=128, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))