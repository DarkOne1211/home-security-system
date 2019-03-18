from Crypto.Cipher import AES
from Crypto import Random
import base64

class AESCipher:
    def __init__(self, key, bs):
        self.key = key # secret key
        self.bs = bs   # block size

    def encrypt(self, raw):
        if(len(raw) % self.bs != 0):
            raw = self.pad(raw)
        initVect = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, initVect)
        return base64.b64encode(initVect + cipher.encrypt(raw))

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        initVect = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, initVect)
        return self.unpad(cipher.decrypt(enc[AES.block_size:]))

    # Adds padding if the input isn't a multiple of block size
    def pad(self, raw):
        inputlen = len(raw)
        difference = (self.bs - inputlen) % self.bs
        raw +=  difference.to_bytes(difference, 'big')
        return raw

    # Removes the padding
    def unpad(self, s):
        return s[:-ord(s[len(s)-1:])]


if __name__ == "__main__":
    #img = cv2.imread('../siamese_network_facial_recognition/recorded_images/praveen_seeniraj/testImage0.png')
    msg = b'testtes'
    #msg = bytes(img)
    # create the encryption object
    cipher = AESCipher('mysecretpassword', 16)
    # encrpyt
    encrypted = cipher.encrypt(msg)
    print(len(encrypted))
    print(type(encrypted))
    # decrypt
    decrypted = cipher.decrypt(encrypted)

    #print(encrypted)
    print(decrypted == msg)
