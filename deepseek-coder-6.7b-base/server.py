import socket
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
from transformers import pipeline


def server(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = (host, port)
    server_socket.bind(server_address)

    server_socket.listen(1)

    print("Server started.")

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="balanced")

    code_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt', pad_token_id=tokenizer.eos_token_id)


    print("Server is listening...")

    while True:
        client_socket, client_address = server_socket.accept()

        try:
            prompt = client_socket.recv(8192)
            prompt = prompt.decode()

            if prompt == "exit":
                client_socket.close()
                break

            output = code_generator(prompt, max_length=8192)[0]['generated_text']

            client_socket.sendall(output.encode())
        finally:
            client_socket.close()

    server_socket.close()

def main():
    parser = argparse.ArgumentParser(description="Server for DeepSeekCoder 6.7b Base")
    parser.add_argument("--host", type=str, help="Host to bind the server to", default="localhost")
    parser.add_argument("--port", type=int, help="Port to bind the server to", default=54321)

    args = parser.parse_args()
    host = args.host
    port = args.port

    server(host=host, port=port)

if __name__ == "__main__":
    main()
