from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import pipeline

custom_cache_dir = "/data/volume_2"
token = "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"
model_name = "deepseek-ai/deepseek-coder-33b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token)

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir, token=token, torch_dtype=torch.bfloat16, device_map="sequential")

code_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt', pad_token_id=tokenizer.eos_token_id)

input_string = "Write a quicksort algorithm in python."
generated_code = code_generator(input_string, max_length=1000)
print(generated_code)
print(generated_code[0]['generated_text'])


# from vllm import LLM, SamplingParams


# custom_cache_dir = "/data/volume_2"
# tp_size = 1 # Tensor Parallelism
# sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=320)
# model_name = "deepseek-ai/deepseek-coder-6.7b-base"
# llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=tp_size, cache_dir=custom_cache_dir)

# prompts = [
#     "If everyone in a country loves one another,",
#     "The research should also focus on the technologies",
#     "To determine if the label is correct, we need to"
# ]
# outputs = llm.generate(prompts, sampling_params)

# generated_text = [output.outputs[0].text for output in outputs]
# print(generated_text)

# # deepseek-coder-6.7b-base

# import time

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# device0 = torch.device("cuda:0")
# device1 = torch.device("cuda:1")

# start_time = time.time()
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", local_files_only=True)
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", local_files_only=True, torch_dtype=torch.
# bfloat16)
# model = model.to(device0)

# print(f"Loading model and tokiner took: {time.time() - start_time} seconds")

# input_text = "#write a quick sort algorithm. Only provide the code with no additional text"
# inputs = tokenizer(input_text, return_tensors="pt").to(device0)
# outputs = model.generate(**inputs, max_length=128)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# print(f"Total time taken: {time.time() - start_time} seconds")

# # deepseek-coder-6.7b-instruct

# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16)
# model = model.to(device1)
# messages=[
#     { 'role': 'user', 'content': "write a quick sort algorithm in python. Only provide the code with no additional text"}
# ]

# inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device1)
# # tokenizer.eos_token_id is the id of <|EOT|> token
# outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
# print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))


# while True:
#     pass