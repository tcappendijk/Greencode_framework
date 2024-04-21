import socket
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def start_deepseek_coder():

    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.
    bfloat16).cuda()

    print(f"Loading model and tokiner took: {time.time() - start_time} seconds")

    return tokenizer, model

def main(tokenizer, model):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to a port
    server_address = ('localhost', 12345)  # localhost and port 12345
    server_socket.bind(server_address)

    # Listen for incoming connections
    server_socket.listen(1)

    print("Server is listening...")

    while True:
        # Wait for a connection
        client_socket, client_address = server_socket.accept()

        try:
            print("Connection from", client_address)

            # Receive data from the client
            prompt = client_socket.recv(1024)
            prompt = prompt.decode()
            print("Received prompt:", prompt)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_length=128)
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Send data to the client
            client_socket.sendall(output.encode())
        finally:
            # Clean up the connection
            client_socket.close()

if __name__ == "__main__":
    tokenizer, model = start_deepseek_coder()

    print("Deepseek-coder is ready to serve!")
    main(tokenizer, model)
