import asyncio
from threading import Thread
from yolo import yolo
import cv2  # requires python-opencv
from queue import Queue
from tello_asyncio import Tello, VIDEO_URL

print("[main thread] START")

# Global
RECORDING = True
CENTER = 480, 300, 300
command_queue = Queue()

###############################################
###     Drone Control in Worker Thread      ###
###############################################

def fly(command_queue):
    print("[fly thread] START")
    async def check_commands(drone):
        while True:
            if not command_queue.empty():
                command = command_queue.get(block=False)
                if command == 't_l':
                    await drone.turn_counterclockwise(3)
                elif command == 't_r':  # Corrected to 't_r' for turning right
                    await drone.turn_clockwise(3)
                else:
                    await drone.turn_clockwise(0)
                print(f"[fly thread] Processing command: {command}")
            await asyncio.sleep(0.1)  # Small delay to prevent tight loop

    async def main():
        global RECORDING
        global CENTER
        vertical = forwardBackward = rotation = 0
        drone = Tello()
        try:
            await asyncio.sleep(1)
            await drone.connect()
            await drone.start_video(connect=False)
            await drone.takeoff()
            await drone.set_speed(10)
            await drone.move_up(80)
            while RECORDING:  # Use RECORDING to control the loop
                x, y, w = CENTER
                if x > 530:
                    rotation = +30           
                elif x < 430:
                    rotation = -30
                else:
                    rotation = 0
                
                # Y movement logic #
                # if y > 350: 
                #     vertical = -20 
                # elif y < 250: 
                #     vertical = 20 
                # else:
                #     vertical = 0
                    
                if w > 500:
                    forwardBackward = -15
                elif w < 300:
                    forwardBackward = 15
                else:
                    forwardBackward = 0
                await drone.remote_control(0, forwardBackward, 0, rotation)
                await asyncio.sleep(1)
        finally:
            await drone.land()
            await drone.stop_video()
            await drone.disconnect()

    asyncio.run(main())

    print("[fly thread] END")

fly_thread = Thread(target=fly, daemon=True, args=(command_queue,))
fly_thread.start()

#####################################################
###     Video capture and GUI in main thread      ###
#####################################################

NEW_VIDEO_URL = VIDEO_URL + '?fifo_size=65536&overrun_nonfatal=1'

print(f"[main thread] OpenCV capturing video from {NEW_VIDEO_URL}")

capture = cv2.VideoCapture(NEW_VIDEO_URL)
capture.open(NEW_VIDEO_URL)
grabbed, test_frame = capture.read()
if grabbed:
    model = yolo(test_frame)
try:
    asyncio.sleep(3)  # Wait for drone to start video stream
    while True:
        grabbed, frame = capture.read()
        if grabbed:
            class_ids, confidences, boxes = model.get_boxes(frame)

            if len(boxes) > 0:
                x, y, w, h = boxes[0]
                print(f'x = {x}, y = {y}, w = {w}, h = {h}')
                CENTER = (x + x/2), (y + h/2), w
            else:
                CENTER = 480, 300, 300

            # Display frame with bounding boxes
            frame = model.box_frame(frame, class_ids, confidences, boxes)
            cv2.imshow("tello-asyncio", frame)
        if cv2.waitKey(1) != -1:
            break
except KeyboardInterrupt:
    pass
finally:
    RECORDING = False
    fly_thread.join()
    if capture:
        capture.release()
    cv2.destroyAllWindows()

print("[main thread] END")