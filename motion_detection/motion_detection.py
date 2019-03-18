from gpiozero import MotionSensor

hcsr501 = MotionSensor(4)

print("Motion Detection Started")

while (True):
    print("no motion")
    hcsr501.wait_for_motion()
    if hcsr501.motion_detected:
        print("motion detected")
