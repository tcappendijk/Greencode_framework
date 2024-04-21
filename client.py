import socket

def main():
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    server_address = ('localhost', 12345)  # localhost and port 12345
    client_socket.connect(server_address)

    try:
        # Send data to the server
        message = "Hello, server!"
        client_socket.sendall(message.encode())

        # Receive data from the server
        data = client_socket.recv(1024)
        print("Received from server:", data.decode())
    finally:
        # Clean up the connection
        client_socket.close()

if __name__ == "__main__":
    main()
