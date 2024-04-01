import asyncio
from threading import Thread
from yolo import yolo
import cv2  # requires python-opencv

from tello_asyncio import Tello, VIDEO_URL

print("[main thread] START")

###############################################
###                                         ###
###     Drone Control in Worker Thread      ###
###                                         ###
###############################################

def fly():
    print("[fly thread] START")

    async def main():
        drone = Tello()
        try:
            await drone.connect()
            await drone.takeoff()
            while True:
                await drone.move_up(20)
                await drone.move_down(20)
        finally:
            await drone.land()
            await drone.stop_video()
            await drone.disconnect()

    # Python 3.7+
    asyncio.run(main())

    print("[fly thread] END")


# # needed for drone.wifi_wait_for_network() in worker thread in Python < 3.8
# asyncio.get_child_watcher()

fly_thread = Thread(target=fly, daemon=True)
fly_thread.start()

#####################################################
###                                               ###
###     Video capture and GUI in main thread      ###
###                                               ###
#####################################################

print(f"[main thread] OpenCV capturing video from {VIDEO_URL}")

capture = cv2.VideoCapture(VIDEO_URL)
grabbed, test_frame = capture.read()
if grabbed:
    model = yolo(test_frame)

try:
    while True:
        grabbed, frame = capture.read()
        frame = model.box_frame(frame)
        if grabbed:
            cv2.imshow("tello-asyncio", frame)

        if cv2.waitKey(1) != -1:
            break
except KeyboardInterrupt:
    pass
finally:
    # tidy up
    if capture:
        capture.release()
    cv2.destroyAllWindows()

print("[main thread] END")