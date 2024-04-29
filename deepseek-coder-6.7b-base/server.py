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
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.bfloat16)

    # Check for the gpu with the most free memory
    gpu_index = get_gpu_with_most_free_memory()
    if gpu_index is None:
        return None, None, None

    gpu_device = torch.device(f"cuda:{gpu_index}")
    model = model.to(gpu_device)

    return tokenizer, model, gpu_device


def server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = ('localhost', 54321)
    server_socket.bind(server_address)

    server_socket.listen(1)

    print("Server started.")

    tokenizer, model, gpu_device = initialize_model()

    if tokenizer is None or model is None or gpu_device is None:
        print("Error occurred while initializing the model.")
        return

    print("Server is listening...")

    while True:
        client_socket, client_address = server_socket.accept()

        try:
            print("Connection from", client_address)

            prompt = client_socket.recv(1024)
            prompt = prompt.decode()
            print("Received prompt:", prompt)

            if prompt == "exit":
                client_socket.close()
                break

            inputs = tokenizer(prompt, return_tensors="pt").to(gpu_device)
            outputs = model.generate(**inputs, max_new_tokens=1024, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)

            output = tokenizer.decode(outputs[0][len(inputs[0]):])
            # output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            client_socket.sendall(output.encode())
        finally:
            client_socket.close()

    server_socket.close()

def main():
    server()

if __name__ == "__main__":
    main()


# import asyncio
# import socket
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import subprocess

# tokenizer = None
# model = None
# gpu_device = None

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

# tokenizer, model, gpu_device = initialize_model()

# async def handle_client(reader, writer):
#     client_address = writer.get_extra_info('peername')
#     print("Connection from", client_address)

#     try:
#         prompt = await reader.read(1024)
#         prompt = prompt.decode()
#         print("Received prompt:", prompt)

#         if prompt == "exit":
#             writer.close()
#             await writer.wait_closed()

#             # This is not the best way to exit the server, but it works for now. I cannot find a way to not busy wait for the server to close.
#             exit(0)

#         messages = [{'role': 'user', 'content': prompt}]


#         if tokenizer is None or model is None or gpu_device is None:
#             print("Error occurred while initializing the model.")
#             return

#         inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(gpu_device)
#         outputs = model.generate(inputs, max_new_tokens=1024, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)

#         output = tokenizer.decode(outputs[0][len(inputs[0]):])

#         writer.write(output.encode())
#         await writer.drain()
#     finally:
#         writer.close()

# async def main():
#     server = await asyncio.start_server(handle_client, 'localhost', 12345)

#     async with server:
#         print("Server is listening...")
#         await server.serve_forever()

# if __name__ == "__main__":
#     asyncio.run(main())
