import os
import sys
import json
import bluetooth
import numpy as np

from PIL import Image
from pprint import pprint

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

class Station:

    def __init__(self):
        # Load config
        with open('config_server.json', 'r') as fp:
            config_vars = json.load(fp)

        self.HOST = config_vars['host']
        self.PORT = config_vars['port']

    def unwrap_payload(self, message):
        header = message[:3]
        payload = message[3:]

        # Unpacking header
        message_length = int.from_bytes(header, 'big') & 0xFFFFF
        header_title = int.from_bytes(header[:1], 'big') >> 4

        rcpt_type   = "base" if (header_title & 0x08) == 0x08 else "pkg"
        pkg_type    = "door" if (header_title & 0x04) == 0x04 else "window"
        encrypted   = (header_title & 0x02) == 0x02
        image       = (header_title & 0x01) == 0x01

        # The full message wasn't recieved
        if len(payload) < message_length:
            # -- read rest of the payload and attempt again
            raise OverflowError(json.dumps({
                'Expected': message_length,
                'Received': len(payload),
                'Remaining': message_length - len(payload)
            }))

        # Some kind of mismatch -- corruption/hacking attempt
        if len(payload) > message_length:
            raise ValueError("Inconsistent payload and header")

        # Decrypt payload if necessary
        if encrypted:
            pass    # TODO

        # Payload is an update
        if not image:
            payload = payload.decode()

        # Payload is an image
        else:
            image_w = int.from_bytes(payload[:2], 'big')
            image_h = int.from_bytes(payload[2:4], 'big')

            payload = np.array(list(payload[4:]), dtype=np.uint8) \
                        .reshape(image_w, image_h)

            payload = Image.fromarray(payload, 'L')

        return {
            'header': {
                'rcpt_type': rcpt_type,
                'pkg_type': pkg_type,
                'encrypted': encrypted,
                'image': image,
                'length': message_length
            },
            'payload': payload
        }


    def start(self):
        # Opening comms to all clients
        s = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        s.bind(("", self.PORT))
        s.listen(1)

        print("Starting server at {0}:{1}".format(self.HOST, self.PORT))

        while True:
            conn, addr = s.accept() # Main thread just creates connections for new clients

            if os.fork() == 0:      # Child threads does handles comms for each client
                print('Connected by', addr)
                # Event loop
                while True:
                # Comm data from client
                    data = b''
                    packetSize = conn.recv(4) # Receive data payload size
                    self.SIZE = int.from_bytes(packetSize,'big')
                    while(len(data) < self.SIZE):
                    	data += conn.recv(self.SIZE - len(data))
                    if not data:
                        break

                    try:
                        message = self.unwrap_payload(data)   # TODO - handle OverflowError
                    except OverflowError as oerr:
                        sizes = json.loads(str(oerr))
                        data += conn.recv(sizes['Remaining'])
                        pass

                    message = self.unwrap_payload(data)
                    pprint(message['header'])
                    with open('sample_recvd_image.png', 'wb') as imf:
                        message['payload'].save(imf)

                    # conn.sendall(b"-- Received image --")
                    # conn.sendall(message['payload'][::-1].encode())
                    # Exit thread
                conn.close()
                exit(0)         # Exit child thread if connection is lost with client
        s.close()

if __name__ == '__main__':
    Station().start()
