from djitellopy import Tello

tello = Tello()

tello.connect()
tello.takeoff()

tello.rotate_clockwise(20)
import time
time.sleep(3)
#tello.move_left(100)
#tello.rotate_counter_clockwise(90)
#tello.move_forward(100)


#while true:
    #print("hello")

tello.land()

