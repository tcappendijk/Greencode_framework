import socket
import argparse

def client(prompt: str):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = ('localhost', 12345)  # localhost and port 12345
    client_socket.connect(server_address)

    try:
        prompt = input("Enter a prompt: ")
        client_socket.sendall(prompt.encode())

        data = client_socket.recv(1024)
        print("Received from server:", data.decode())
    finally:
        client_socket.close()

def main():
    parser = argparse.ArgumentParser(description="Client for DeepSeekCoder 6.7b Instruct")
    parser.add_argument("--prompt", type=str, help="Prompt to send to the server")
    args = parser.parse_args()

    prompt = args.prompt

    if prompt:
        client(prompt)
    else:
        print("Prompt is required. Use --prompt argument to specify the prompt.")
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()