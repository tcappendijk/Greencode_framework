# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from torch.nn.parallel import DataParallel

# # Load your model
# model_name = "../CodeLlama-70b-Python-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()

# # Check if CUDA is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Move your model to the device
# model = model.to(device)

# # Wrap the model with DataParallel
# model = DataParallel(model)

# input_text = "#write a quick sort algorithm"
# inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
# outputs = model.generate(**inputs, max_length=128)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# while True:
#     pass

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

token = "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"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-Instruct-hf", token=token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-Instruct-hf", token=token)