from djitellopy import Tello

import time

tello = Tello()

tello.connect()
tello.takeoff()
tello.move_up(80)
print(f'Battery: {tello.get_battery()}%')

tello.streamon()
time.sleep(2)