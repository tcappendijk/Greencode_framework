import socket
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse


def server(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = (host, port)
    server_socket.bind(server_address)

    server_socket.listen(1)

    print("Server started.")

    custom_cache_dir = "/data/volume_2"
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, cache_dir=custom_cache_dir)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=custom_cache_dir).cuda()

    print("Server is listening...")

    max_length = 2048

    while True:
        client_socket, client_address = server_socket.accept()

        try:
            prompt = client_socket.recv(max_length)
            prompt = prompt.decode()

            if prompt == "exit":
                client_socket.close()
                break

            messages=[
                { 'role': 'user', 'content': prompt}
            ]

            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

            outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
            output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

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
