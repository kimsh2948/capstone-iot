import socket

host = ''
port = 9999

server_sock = socket.socket(socket.AF_INET)
server_sock.bind((host, port))
server_sock.listen(1)

print("...listen")
client_sock, addr = server_sock.accept()

print('Connected by', addr)
data = client_sock.recv(1024)
print(data.decode("utf-8"), len(data))

data2 = int(input("send data: "))
print(data2.encode())
client_sock.send(data)
client_sock.send(data2.to_bytes(4, byteorder='little'))

client_sock.close()
server_sock.close()
