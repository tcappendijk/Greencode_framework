from transformers import AutoTokenizer, AutoModelForCausalLM

custom_cache_dir = "/data/volume_2"

token = "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"
model_name = "meta-llama/CodeLlama-70b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, trust_remote_code=True)