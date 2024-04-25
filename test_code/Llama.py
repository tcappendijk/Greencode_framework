# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-13b-Instruct-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/CodeLlama-13b-Instruct-hf")