# import socket
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import subprocess


# def get_gpu_with_most_free_memory():
#     try:
#         result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"])

#         gpu_info = result.decode().strip().split('\n')
#         gpu_info = [info.split(', ') for info in gpu_info]
#         gpu_info = [(int(index), int(free_memory), int(total_memory)) for index, free_memory, total_memory in gpu_info]
#         gpu_info.sort(key=lambda x: x[1], reverse=True)
#         gpu_index = gpu_info[0][0]

#         return gpu_index
#     except Exception as e:
#         print(f"Error occurred while getting GPU information: {e}")
#         return None


# def initialize_model():
#     tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16)

#     # Check for the gpu with the most free memory
#     gpu_index = get_gpu_with_most_free_memory()
#     if gpu_index is None:
#         return None, None, None

#     gpu_device = torch.device(f"cuda:{gpu_index}")
#     model = model.to(gpu_device)

#     return tokenizer, model, gpu_device


# def server():
#     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#     server_address = ('localhost', 12345)
#     server_socket.bind(server_address)

#     server_socket.listen(1)

#     print("Server started.")

#     tokenizer, model, gpu_device = initialize_model()

#     if tokenizer is None or model is None or gpu_device is None:
#         print("Error occurred while initializing the model.")
#         return

#     print("Server is listening...")

#     while True:
#         client_socket, client_address = server_socket.accept()

#         try:
#             print("Connection from", client_address)

#             prompt = client_socket.recv(1024)
#             prompt = prompt.decode()
#             print("Received prompt:", prompt)

#             if prompt == "exit":
#                 break

#             messages=[
#                 { 'role': 'user', 'content': prompt}
#             ]

#             inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(gpu_device)
#             outputs = model.generate(inputs, max_new_tokens=1024, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)

#             output = tokenizer.decode(outputs[0][len(inputs[0]):])

#             client_socket.sendall(output.encode())
#         finally:
#             client_socket.close()

#     server_socket.close()

# def main():
#     server()

# if __name__ == "__main__":
#     main()


import asyncio
import socket
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import subprocess

def get_gpu_with_most_free_memory():
    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"])

        gpu_info = result.decode().strip().split('\n')
        gpu_info = [info.split(', ') for info in gpu_info]
        gpu_info = [(int(index), int(free_memory), int(total_memory)) for index, free_memory, total_memory in gpu_info]
        gpu_info.sort(key=lambda x: x[1], reverse=True)
        gpu_index = gpu_info[0][0]

        return gpu_index
    except Exception as e:
        print(f"Error occurred while getting GPU information: {e}")
        return None

def initialize_model():
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16)

    # Check for the gpu with the most free memory
    gpu_index = get_gpu_with_most_free_memory()
    if gpu_index is None:
        return None, None, None

    gpu_device = torch.device(f"cuda:{gpu_index}")
    model = model.to(gpu_device)

    return tokenizer, model, gpu_device

async def handle_client(reader, writer, tokenizer, model, gpu_device):
    client_address = writer.get_extra_info('peername')
    print("Connection from", client_address)

    try:
        prompt = await reader.read(1024)
        prompt = prompt.decode()
        print("Received prompt:", prompt)

        if prompt == "exit":
            return

        messages = [{'role': 'user', 'content': prompt}]


        if tokenizer is None or model is None or gpu_device is None:
            print("Error occurred while initializing the model.")
            return

        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(gpu_device)
        outputs = model.generate(inputs, max_new_tokens=1024, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)

        output = tokenizer.decode(outputs[0][len(inputs[0]):])

        writer.write(output.encode())
        await writer.drain()
    finally:
        writer.close()

async def main():
    tokenizer, model, gpu_device = initialize_model()
    server = await asyncio.start_server(handle_client, 'localhost', 12345, tokenizer, model, gpu_device)

    async with server:
        print("Server started.")
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
