from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.distributed as dist

# Initialize distributed training
torch.distributed.init_process_group(backend='nccl')

# Set the device
device = torch.device("cuda")

# Set the number of processes and rank
world_size = torch.distributed.get_world_size()
rank = torch.distributed.get_rank()

# Initialize distributed training only if it's not already initialized
if not dist.is_initialized():
    # Set environment variable to specify CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)  # Assign different GPU to each rank

    # Initialize distributed training
    dist.init_process_group(backend='nccl')

# Set the device
device = torch.device("cuda")

# Set up the model and tokenizer
custom_cache_dir = "/data/volume_2"
token = "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"
model_name = "meta-llama/CodeLlama-7b-Instruct-hf"

# Ensure each process loads the model independently
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, torch_dtype=torch.bfloat16)
model.to(device)

# Wrap the model with DistributedDataParallel
model = DDP(model)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token)


# # Create a pipeline
# code_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, num_workers=8, device=0, framework='pt', max_length=1000, pad_token_id=tokenizer.eos_token_id)

# # Generate code for an input string
# input_string = "Write a python function to calculate the factorial of a number"
# generated_code = code_generator(input_string, max_length=100)[0]['generated_text']
# print(generated_code)


# device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

# if torch.cuda.is_available():
#     model = DDP(model)
#     print("Model is now DDP")
#     # model = DataParallel(model, device_ids=device_ids)
#     # model = model.cuda()  # Move model to GPU
#     # print("Model is now DataParallel")

# input_text = "Write a quick sort algorithm without test cases. Name the function quick_sort"

# input_ids = tokenizer.encode(input_text, return_tensors="pt")

# if torch.cuda.is_available():
#     input_ids = input_ids.cuda()

# print(input_ids)

# with torch.no_grad():
#     if torch.cuda.is_available():
#         output = model.module.generate(input_ids.cuda(), max_new_tokens=1200, pad_token_id=tokenizer.eos_token_id)
#     else:
#         output = model.generate(input_ids, max_new_tokens=1200, pad_token_id=tokenizer.eos_token_id)

# output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# print(output_text)
