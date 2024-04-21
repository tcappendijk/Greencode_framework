import socket
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def start_deepseek_coder_base_6b():

    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.
    bfloat16).cuda()

    print(f"Loading model and tokiner took: {time.time() - start_time} seconds")
    print("Deepseek-coder-6.7b-base is ready to serve!")

    return tokenizer, model, "deepseek-coder-6.7b-base"


# def start_deepseek_coder_base_33b():

#     start_time = time.time()
#     tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-33b-base", trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-33b-base", trust_remote_code=True, torch_dtype=torch.
#     bfloat16).cuda()

#     print(f"Loading model and tokiner took: {time.time() - start_time} seconds")
#     print("Deepseek-coder-33b-base is ready to serve!")

#     return tokenizer, model, "deepseek-coder-33b-base"


def main(llms):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to a port
    server_address = ('localhost', 12345)
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

            code_output = ""

            # Generate code completions
            for llm in llms:
                tokenizer, model, name = llm
                output = ""
                try:
                    input = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True, padding="max_length").input_ids.cuda()
                    output = model.generate(input, max_length=1024, num_return_sequences=1, pad_token_id=50256, eos_token_id=50256)[0]
                    output = tokenizer.decode(output, skip_special_tokens=True)
                    code_output += f"Code completion using {name}:\n{output}\n"
                except Exception as e:
                    print(f"Error while generating code completion using {name}: {e}")

            # Send data to the client
            client_socket.sendall(code_output.encode())
        finally:
            # Clean up the connection
            client_socket.close()

if __name__ == "__main__":

    llms = []
    tokenizer, model, name = start_deepseek_coder_base_6b()
    llms.append((tokenizer, model, name))

    # tokenizer, model, name = start_deepseek_coder_base_33b()
    # llms.append((tokenizer, model, name))
    main(llms)
