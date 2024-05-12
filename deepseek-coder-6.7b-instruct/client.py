import socket
import argparse

def client(prompt: str, host, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = (host, port)
    client_socket.connect(server_address)

    try:
        client_socket.sendall(prompt.encode())

        data = client_socket.recv(1024)

        # This print is necessary to be able to read the output from stdout
        print(data.decode())
    finally:
        client_socket.close()

def main():
    parser = argparse.ArgumentParser(description="Client for DeepSeekCoder 6.7b Instruct")
    parser.add_argument("--prompt", type=str, help="Prompt to send to the server")
    parser.add_argument("--host", type=str, help="Host to connect to", default="localhost")
    parser.add_argument("--port", type=int, help="Port to connect to", default=12345)
    args = parser.parse_args()

    prompt = args.prompt
    host = args.host
    port = args.port

    if prompt:
        client(prompt, host, port)
    else:
        print("Prompt is required. Use --prompt argument to specify the prompt.")
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()