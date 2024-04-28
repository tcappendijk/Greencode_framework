# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("CodeLlama-7b-Instruct-hf")
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("CodeLlama-7b-Instruct-hf", torch_dtype=torch.bfloat16).cuda()

input_text = "#write a quick sort algorithm"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

while True:
    pass