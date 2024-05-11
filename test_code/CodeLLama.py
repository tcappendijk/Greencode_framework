# from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# from torch.nn.parallel import DistributedDataParallel

num_devices = torch.cuda.device_count()

# Print available CUDA devices and set one
for i in range(num_devices):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# custom_cache_dir = "/data/volume_2"

# token = "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"
# model_name = "meta-llama/CodeLlama-7b-Instruct-hf"

# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, trust_remote_code=True)

# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, trust_remote_code=True, torch_dtype=torch.bfloat16)

# if torch.cuda.is_available():
#     model = model.cuda()
#     model = DistributedDataParallel(model)

# input_text = "#write a quick sort algorithm. Provide only the code. Name the function quick_sort"

# input_ids = tokenizer.encode(input_text, return_tensors="pt")

# if torch.cuda.is_available():
#     input_ids = input_ids.cuda()

# with torch.no_grad():
#     if torch.cuda.is_available():
#         output = model.module.generate(input_ids, max_new_tokens=1200)
#     else:
#         output = model.generate(input_ids, max_new_tokens=1200)

# output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# print(output_text)
