# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("CodeLlama-13b-Instruct-hf")
model = AutoModelForCausalLM.from_pretrained("CodeLlama-13b-Instruct-hf")