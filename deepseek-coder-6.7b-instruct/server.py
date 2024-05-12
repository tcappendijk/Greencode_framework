import socket
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import subprocess
import argparse
from transformers import pipeline


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


def server(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = (host, port)
    server_socket.bind(server_address)

    server_socket.listen(1)

    print("Server started.")

    # tokenizer, model, gpu_device = initialize_model()

    # if tokenizer is None or model is None or gpu_device is None:
    #     print("Error occurred while initializing the model.")
    #     return

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="balanced")

    code_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt', pad_token_id=tokenizer.eos_token_id)

    print("Server is listening...")

    while True:
        client_socket, client_address = server_socket.accept()

        try:
            prompt = client_socket.recv(1024)
            prompt = prompt.decode()

            if prompt == "exit":
                client_socket.close()
                break

            messages=[
                { 'role': 'user', 'content': prompt}
            ]

            # inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(gpu_device)
            # outputs = model.generate(inputs, max_new_tokens=1024, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)

            # output = tokenizer.decode(outputs[0][len(inputs[0]):])
            output = code_generator(prompt, max_length=1000)[0]['generated_text']

            client_socket.sendall(output.encode())
        finally:
            client_socket.close()

    server_socket.close()

def main():
    parser = argparse.ArgumentParser(description="Server for DeepSeekCoder 6.7b Instruct")
    parser.add_argument("--host", type=str, help="Host to bind the server to", default="localhost")
    parser.add_argument("--port", type=int, help="Port to bind the server to", default=12345)

    args = parser.parse_args()
    host = args.host
    port = args.port

    server(host=host, port=port)

if __name__ == "__main__":
    main()
