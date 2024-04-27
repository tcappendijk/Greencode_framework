# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("CodeLlama-13b-Instruct-hf")
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("CodeLlama-13b-Instruct-hf", torch_dtype=torch.bfloat16)