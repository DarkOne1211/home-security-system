# Facial Recognition on the Raspberry Pi

## SETUP
- Setup the raspberry pi according to the instructions in this [link](https://www.raspberrypi.org/downloads/) (This is the default raspbian os setup if you already have it skip this step)
  - Do not use **NOOBS** use [**Raspbian**](https://www.raspberrypi.org/downloads/raspbian/)
- Follow the steps to connect the camera from [here](https://www.raspberrypi.org/documentation/usage/camera/)
- Bootup your raspberry pi
- Clone this repo
- Run ``` sudo sh ./setup.sh```
- Type in your sudo password and **ALL** requirements will automatically 
install.
    - You have to be stupid to fuck this up. Yeet.

## Usage
- All the functions are stored inside ***facial_recognition_helper.py***
- This module can be imported like any other python module using 
```python
import facial_recognition_helper
```
- Before using any other function run the **initialize()** function
- Once initialized the **add_user(name)** function can be called with a string as an arguemnt
    - Upon running the add_user() function the camera will turn on for 10 seconds and record and store the user's facial features
- Once all the users have been added ther person can call train_knn() to train the model that classifies their faces
- Once the model has been trained the user can then call recognize_face() to run facial detection

```python
# Example Code:
from facial_recognition_helper import captureImages
from facial_recognition_helper import extract_align_faces
from facial_recognition_helper import initialize
from facial_recognition_helper import add_user
from facial_recognition_helper import extract_features
from facial_recognition_helper import train_knn
from facial_recognition_helper import recognize_face

# Run each step individually

# Step 1
initialize()
add_user("praveen_seeniraj")
add_user("anand_palanisamy")
add_user("koustav_samaddar")
add_user("hridesh_sainani")

#Step 2
train_knn()

# Step 3 (The actual facial recognition call)
result = recognize_face()
print(result)
```