from socket import socket

client_socket = socket()

ip = '127.0.0.1'
port = 3000

client_socket.connect((ip, port))

data = 'input : '
client_socket.send(data.encode('utf-8'))

data = client_socket.recv(1024).decode('utf-8')
print(data)
