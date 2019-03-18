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
- This module can be imported using

```python
from tone_generator import ToneGenerator
```

- Example Usage

```python
generator = ToneGenerator()
 
frequency_start = 5000        # Frequency to start the sweep from
frequency_end = 10000       # Frequency to end the sweep at
num_frequencies = 200       # Number of frequencies in the sweep
amplitude = 0.50            # Amplitude of the waveform
step_duration = 0.43        # Time (seconds) to play at each step
 
for frequency in numpy.logspace(math.log(frequency_start, 10),
                                math.log(frequency_end, 10),
                                num_frequencies):
 
    print("Playing tone at {0:0.2f} Hz".format(frequency))
    generator.play(frequency, step_duration, amplitude)
    while generator.is_playing():
        pass                # Do something useful in here (e.g. recording)
```