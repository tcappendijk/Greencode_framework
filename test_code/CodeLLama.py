# from transformers import AutoTokenizer, LlamaForCausalLM
# import torch
# from transformers import pipeline
# import argparse

# def generate_code(prompt, model_name="meta-llama/CodeLlama-70b-Instruct-hf", temperature=0.2, top_p=0.95, max_length=512):
#     custom_cache_dir = "/data/volume_2"
#     token = "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"

#     tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, truncation=True)

#     model = LlamaForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, torch_dtype=torch.bfloat16, device_map="sequential")

#     code_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt', pad_token_id=tokenizer.eos_token_id)

#     generated_code = code_generator(prompt, do_sample=True, temperature=temperature, top_p=top_p, truncation=True, padding=False, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
#     print(generated_code[0]['generated_text'])

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Code Generation")
#     parser.add_argument("--prompt", type=str, help="Input prompt")
#     parser.add_argument("--temperature", type=str, help="Temperature value of the model")
#     parser.add_argument("--top_p", type=str, help="Top p value of the model")
#     parser.add_argument("--max_length", type=str, help="Maximum sequence length")
#     parser.add_argument("--model_name", type=str, help="Model name")

#     args = parser.parse_args()

#     prompt = args.prompt
#     temperature = args.temperature
#     top_p = args.top_p
#     max_seq_len = args.max_length
#     model_name = args.model_name

#     generate_code(prompt)

from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from transformers import pipeline
import argparse

def generate_code(input_string):
    custom_cache_dir = "/data/volume_2"
    token = "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"
    model_name = "meta-llama/CodeLlama-70b-Instruct-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, truncation=True)

    model = LlamaForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, torch_dtype=torch.bfloat16, device_map="balanced")

    code_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt', pad_token_id=tokenizer.eos_token_id)

    generated_code = code_generator(input_string, do_sample=True, temperature=0.2, top_p=0.95, truncation=True, padding=False, max_length=512, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    print(generated_code[0]['generated_text'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Generation")
    parser.add_argument("input_string", type=str, help="Input string")
    args = parser.parse_args()
    generate_code(args.input_string)