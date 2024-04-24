import socket
import argparse

def client(ip_address, port, prompt: str):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = (ip_address, port)  # localhost and port 12345
    client_socket.connect(server_address)

    try:
        client_socket.sendall(prompt.encode())

        data = client_socket.recv(1024)
        print("Code recieved from server:")
        print(data.decode(), end="")
    finally:
        client_socket.close()

def main():
    parser = argparse.ArgumentParser(description="Client for DeepSeekCoder 6.7b Instruct")
    parser.add_argument("--server", type=str, default="localhost", help="Server IP address")
    parser.add_argument("--port", type=int, default=12345, help="Port number")
    parser.add_argument("--prompt", type=str, help="Prompt to send to the server")
    args = parser.parse_args()

    prompt = args.prompt
    ip_address = args.server
    port = args.port

    if prompt:
        client(ip_address, port, prompt)
    else:
        print("Prompt is required. Use --prompt argument to specify the prompt.")
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()