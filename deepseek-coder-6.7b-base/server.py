import socket
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


def server(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = (host, port)
    server_socket.bind(server_address)

    server_socket.listen(1)

    print("Server started.")

    custom_cache_dir = "/data/volume_2"
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, cache_dir=custom_cache_dir)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, cache_dir=custom_cache_dir).cuda()

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

            inputs = tokenizer(prompt, return_tensors="pt").cuda()
            outputs = model.generate(**inputs, max_new_tokens=512)

            output = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
