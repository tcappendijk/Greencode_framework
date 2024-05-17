from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from transformers import pipeline
import argparse

def generate_code(prompt):
    custom_cache_dir = "/data/volume_2"
    token = "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"
    model_name = "meta-llama/CodeLlama-70b-Instruct-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, truncation=True)

    model = LlamaForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, torch_dtype=torch.bfloat16, device_map="balanced")

    code_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt', pad_token_id=tokenizer.eos_token_id)

    generated_code = code_generator(prompt, do_sample=True, temperature=0.7, top_p=0.95, truncation=True, padding=False, max_length=512, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    print(generated_code[0]['generated_text'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Generation")
    parser.add_argument("--prompt", type=str, help="Input prompt")
    args = parser.parse_args()
    generate_code(args.prompt)