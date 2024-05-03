import asyncio
from threading import Thread
from yolo import yolo
import cv2  # requires python-opencv
import os
from queue import Queue
from tello_asyncio import Tello, VIDEO_URL

print("[main thread] START")

# Global
RECORDING = True
CENTER = 480, 360, 960
INITIAL_TAKEOFF_HEIGHT = 80
DRONE_MOVEMENT = CENTER

###############################################
###     Drone Control in Worker Thread      ###
###############################################


'''
0             480            960
|--------------|--------------| 0
|                             |
|                             |
|                             |
|                             |
|                             | 360
|                             |
|                             |
|                             |
|                             |
|--------------|--------------| 720
ROT LEFT     CENTER     ROT RIGHT
'''

def fly():
    print("[fly thread] START")

    async def main():
        global RECORDING
        global CENTER
        global INITIAL_TAKEOFF_HEIGHT
        vertical = forwardBackward = rotation = 0

        drone = Tello()
        try:
            await drone.connect()
            await drone.start_video(connect=False)
            # await drone.takeoff()
            # await drone.move_up(INITIAL_TAKEOFF_HEIGHT)
            while RECORDING:  # Use RECORDING to control the loop
                mid, y, w = DRONE_MOVEMENT

                if mid > 480:
                    rotation = 4        
                elif mid < 480:
                    rotation = -4
                else:
                    rotation = 0
                
                # Y movement logic
                # if y > 350: 
                #     vertical = -20 
                # elif y < 250: 
                #     vertical = 20 
                # else:
                #     vertical = 0
                    
                if w > 60:
                    forwardBackward = -2
                elif w < 30:
                    forwardBackward = 1
                else:
                    forwardBackward = 0

                # await drone.remote_control(0, forwardBackward, vertical, rotation)
                await asyncio.sleep(0.1)
        finally:
            # await drone.land()
            await drone.stop_video()
            await drone.disconnect()

    asyncio.run(main())

    print("[fly thread] END")

fly_thread = Thread(target=fly, daemon=True)
fly_thread.start()

#####################################################
###     Video capture and GUI in main thread      ###
#####################################################

NEW_VIDEO_URL = VIDEO_URL + '?fifo_size=65536&overrun_nonfatal=1'

print(f"[main thread] OpenCV capturing video from {NEW_VIDEO_URL}")

# OpenCV video writer setup to save the video stream
# OUTPUT_FILE = 'output.avi'
# OUTPUT_IDX = 0
# while os.path.exists(f'{OUTPUT_FILE}_{OUTPUT_IDX}.avi'):
#     OUTPUT_IDX += 1
# video_writer = cv2.VideoWriter(f'{OUTPUT_FILE}_{OUTPUT_IDX}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (960, 720))
# print(f"[main thread] Video writer created: {OUTPUT_FILE}_{OUTPUT_IDX}.avi")

capture = cv2.VideoCapture(NEW_VIDEO_URL)
grabbed, test_frame = capture.read()
if grabbed:
    model = yolo(test_frame)
print(f'[main thread] Test frame grabbed with shape: {test_frame.shape}')

frame_count = 0
try:
    while True:
        grabbed, frame = capture.read()
        if grabbed:
            if frame_count % 3 == 0:
                class_ids, confidences, boxes = model.get_boxes(frame)
                frame = model.box_frame(frame, class_ids, confidences, boxes)
            
            # if len(boxes) > 0:
            #     x, y, w, h = boxes[0]
            #     print(f'x = {x}, y = {y}, w = {w}, h = {h}')
            #     DRONE_MOVEMENT = (x+w)/2, y, w
            # else:
            #     mid, y, w = DRONE_MOVEMENT
            #     DRONE_MOVEMENT = mid, y, 960 # Keeps previous values but stops drone from moving

            cv2.imshow("tello-asyncio", frame)
        if cv2.waitKey(1) != -1:
            break
        frame_count += 1

except KeyboardInterrupt:
    pass
finally:
    RECORDING = False
    fly_thread.join()
    if capture:
        capture.release()
    cv2.destroyAllWindows()

print("[main thread] END")