import sys
import json
import socket
import numpy as np
from PIL import Image
sys.path.insert(0, '../crypto')
from AESCipher import AESCipher as AES
'''
    Protocol:
        Header [24 bits]
        Body [n bytes]

        Header [24 bits]:
            - Actor Type    [1 bit]:    1 if base, 0 if pkg -- not used now,
                                            but could be used to create mesh networks
            - Client Type   [1 bit]:    1 if door pkg, 0 if window pkg, X for base
            - Encrypted     [1 bit]:    1 if encrypted else 0
            - Message Type  [1 bit]:    1 if image, 0 if update
            - Message Size  [20 bits]:  unsigned integer denoting size in bytes
'''

class Package:

    def __init__(self):
        # Load config
        with open('config_client.json', 'r') as fp:
            config_vars = json.load(fp)

        self.HOST = config_vars['host']
        self.PORT = config_vars['port']
        self.SIZE = config_vars['size']
        self.KEY = config_vars['aeskey']
        self.BS = config_vars['blocksize']
        self.ENCRYPTOR = AES(self.KEY, self.BS)
        # TODO: read from config or constructor
        # For now hard-coding to a door package
        self.pkg_type = 1   # 1 if door, 0 if window


    def wrap_payload(self, payload, encrypted=True):
        header_title = 0x00

        if self.pkg_type == 1:
            header_title = header_title | 0x04

        if encrypted:
            header_title = header_title | 0x02

        # Payload is an update
        if isinstance(payload, str):
            payload = payload.encode()

        # Payload is an image
        elif isinstance(payload, Image.Image):
            header_title = header_title | 0x01
            payload = np.array(payload.convert('L'))
            image_w, image_h = payload.shape
            payload = payload.flatten()

            # Payload sanity testing
            for elem in payload:
                if not isinstance(elem, np.uint8):
                    raise TypeError("Payload values must be uint8")
                elif -1 > elem > 255:
                    raise ValueError("Payload values must be in range of [0, 255]")
            payload = image_w.to_bytes(2, 'big') + image_h.to_bytes(2, 'big') + bytes(payload)
        else:
            raise TypeError("Payload type is invalid")

        # Encryption
        if encrypted:
            payload = self.ENCRYPTOR.encrypt(payload)

        if len(payload) & 0xFFFFF != len(payload):
            raise ValueError("Payload size cannot fit in 20 bits")

        # header = bytes([header_title]) + (len(payload).to_bytes(3, 'big'))
        header = (header_title << 20) | (len(payload))
        header = header.to_bytes(3, 'big')

        return header + payload


    def start(self):
        print("Starting client")

        # Connecting to server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.HOST, self.PORT))
            print("Connected to {0}:{1}".format(self.HOST,self.PORT))

            # Initialisation
            pass    # TODO

            # Communication loop
            # Getting user input
            print('> ', end='')
            t = str(input())

            i = Image.open("sample_image.png")

            # Exit Condition
            if t == 'exit':
                exit()

            # Send message
            # Send update
            # s.sendall(self.wrap_payload(t))

            # Send image
            wrappedPayload = self.wrap_payload(i)
            s.sendall(len(wrappedPayload).to_bytes(4,byteorder='big'))
            s.sendall(wrappedPayload)

            # Receive response
            # data = s.recv(self.SIZE)

            # Act on response data
            # print(data.decode())

        print('Quitting Client')


if __name__ == '__main__':
    Package().start()
