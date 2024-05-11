from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.parallel import DataParallel

# Custom cache directory
custom_cache_dir = "/data/volume_2"

# Token and model name
token = "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"
model_name = "meta-llama/CodeLlama-7b-Instruct-hf"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, trust_remote_code=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, trust_remote_code=True, torch_dtype=torch.bfloat16)

# Move model to CUDA if available and wrap with DataParallel
if torch.cuda.is_available():
    model = model.cuda()
    model = DataParallel(model)

# Input text
input_text = "#write a quick sort algorithm. Provide only the code. Name the function quick_sort"

# Tokenize input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Move input to CUDA if available
if torch.cuda.is_available():
    input_ids = input_ids.cuda()

# Generate output sequence
with torch.no_grad():
    output = model.generate(input_ids)

# Decode output sequence
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
