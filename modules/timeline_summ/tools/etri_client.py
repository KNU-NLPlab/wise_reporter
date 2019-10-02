# -*- coding : utf-8 -*-
# etri_client.py
# It control connection to etri nlp server
import socket
from time import sleep

#
class Etri_client():
    def __init__(self, host_name, port_number, silence=False):
        self.host_name = host_name
        self.port_number = port_number
        self.silence = silence

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error as err:
            return ' '
            # print("Socket creation was failed with err : {}".format(err))

    # connect to server
    def connect(self):
        try :
            self.socket.connect((self.host_name, self.port_number))
            if not self.silence:
                pass
                # print("Success to connect to server")
        except socket.error as err:
            return ' '
            # print("Fail to connect to server with err : {}".format(err))

    def send(self, msg):
        try:
            self.socket.sendall(bytes(msg, 'utf-8'))
            self.socket.shutdown(socket.SHUT_WR) # must exist
            if not self.silence:
                pass
                # print("Success to send msg to server")
        except IOError as err:
            return ' '
            # print("I/O exception: {}".format(err))
            # print("while sending msg: {}".format(msg))
            # exit(err)

    def recv(self, weight=0):
        try:
            data_list = []
            while True:
                # data = self.socket.recv(65536)
                data = self.recv_all(weight)
                if not data:
                    break
                recv_msg = data.decode('utf-8').strip()
                data_list.append(recv_msg)
                # print(recv_msg)
            if not self.silence:
                pass
                # print("Success to receive msg from server")
            return ''.join(data_list)
        except IOError as err:
            return ' '
            # print("I/O exception: {}".format(err))
            # print("while receving msg")
            # return None
        except UnicodeDecodeError:
            return ' '
            # print("UnicodeDecoderError")
            # print(data)
            # print(len(data))

    def recv_all(self, weight=0):
        buf_size = 4096
        data = b""
        while True:
            part = self.socket.recv(buf_size)
            data = b"".join([data, part])
            # print(len(part), data)
            # 서버 과부화? 때문에 너무 빨리 던지면 결과를 잘 못 얻어 옴
            sleep(0.001*(10**weight))
            if len(part) < buf_size:
                break

        return data

    def close_connection(self):
        try:
            self.socket.close()
            if not self.silence:
                pass
                # print("Success to close")
        except socket.error as err:
            return ' '
            # print("Failed to close socket with error {}".format(err))
