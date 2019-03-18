# Facial Recognition on the Raspberry Pi

## SETUP
- Setup the raspberry pi according to the instructions in this [link](https://www.raspberrypi.org/downloads/) (This is the default raspbian os setup if you already have it skip this step)
  - Do not use **NOOBS** use [**Raspbian**](https://www.raspberrypi.org/downloads/raspbian/)
- Bootup your raspberry pi
- Clone this repo
- Run ``` sudo sh ./setup.sh```
- Type in your sudo password and **ALL** requirements will automatically 
install.
    - You have to be stupid to fuck this up. Yeet.

## Usage
- All the functions are stored inside ***AESCipher.py***
- This module can be imported like any other python module using 
```python
import AESCipher
```
- Instantiate the AESCipher(key, block_size) class
    - The **key** can be any value that is a power of 2 upto 8bytes
    - The **block_size** can be any multiple of 16 (Recommended 16 or 32)
- The class has two methods .encrypt(msg) and .decrypt(encrypted_message) which can be invoked to perform their respective activities
    - **.encrpyt(msg)** takes in a byte string as an arguement and returns the encrypted and base64 encoded bytes string
    - **.decrypt(encrypted_msg)** takes in the encrpyted message as an arguement and return the decrpyted message 

```python
# Example Code:
cipher = AESCipher('mysecretpassword', 16) # Key only for example
# encrpyt
encrypted = cipher.encrypt(msg)
# decrypt
decrypted = cipher.decrypt(encrypted)
```