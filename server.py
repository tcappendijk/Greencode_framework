import socket

def main():
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
            data = client_socket.recv(1024)
            print("Received:", data.decode())

            # Echo back the data
            client_socket.sendall(data)
        finally:
            # Clean up the connection
            client_socket.close()

if __name__ == "__main__":
    main()
