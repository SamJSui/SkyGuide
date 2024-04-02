import asyncio
from threading import Thread
from yolo import yolo
import cv2  # requires python-opencv

from tello_asyncio import Tello, VIDEO_URL

print("[main thread] START")

# global
RECORDING = False

###############################################
###                                         ###
###     Drone Control in Worker Thread      ###
###                                         ###
###############################################


# IMPORTANT #
# Currently trying to get the fly thred to gracefully exit with the program
def fly():
    print("[fly thread] START")

    async def main():
        global RECORDING
        drone = Tello()
        try:
            await asyncio.sleep(1)
            await drone.connect()
            await drone.start_video(connect=False)
            RECORDING = True
            await drone.takeoff()

            while True:
                if RECORDING == False:
                    break
                await drone.move_up(20)
                await drone.move_down(20)
                await drone.turn_counterclockwise(30)
                await drone.turn_clockwise(60)
                await drone.turn_counterclockwise(30)
            await drone.land()
        finally:
            await drone.stop_video()
            RECORDING = False
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
capture.open(VIDEO_URL)
grabbed, test_frame = capture.read()
if grabbed:
    model = yolo(test_frame)

try:
    while True:
        grabbed, frame = capture.read()
        if grabbed:
            frame = model.box_frame(frame)
            cv2.imshow("tello-asyncio", frame)

        if cv2.waitKey(1) != -1:
            break
except KeyboardInterrupt:
    pass
finally:
    print('tidy up')
    if capture:
        capture.release()
    cv2.destroyAllWindows()

    # needs further testing
    RECORDING = False
    fly_thread.join()


print("[main thread] END")