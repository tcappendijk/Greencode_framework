from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from transformers import pipeline
import argparse
import sys

def generate_code(prompt, model_name, max_length):
    custom_cache_dir = "/data/volume_2"
    token = "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"

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, truncation=True)

    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, torch_dtype=torch.bfloat16, device_map="balanced")

    code_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt', pad_token_id=tokenizer.eos_token_id)

    generated_code = code_generator(prompt, truncation=True, padding=True, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Redirect stdout to a temporary buffer
    stdout_backup = sys.stdout
    sys.stdout = open('temp.txt', 'w')

    print(generated_code[0]['generated_text'])

    # Restore stdout and read the contents of the temporary buffer
    sys.stdout.close()
    sys.stdout = stdout_backup

    with open('temp.txt', 'r') as f:
        cleaned_output = f.read()

    print(cleaned_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Generation")
    parser.add_argument("--prompt", type=str, help="Input prompt")
    parser.add_argument("--model_name", type=str, help="Model name", default="meta-llama/CodeLlama-70b-Instruct-hf")
    parser.add_argument("--max_length", type=int, help="Maximum length of the generated code", default=512)

    args = parser.parse_args()

    prompt = args.prompt
    max_seq_len = args.max_length
    model_name = args.model_name

    generate_code(prompt, model_name, max_seq_len)