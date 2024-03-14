import cv2
import numpy as np
from djitellopy import Tello
import time

# Initialize Tello drone
tello = Tello()
tello.connect()
print(f'Battery: {tello.get_battery()}%')

# Takeoff and basic movement
tello.takeoff()
tello.move_up(80)

# Start video stream
tello.streamon()
time.sleep(2)

# Load YOLO
net = cv2.dnn.readNet("skyguide/yolo/yolov3.weights", "skyguide/yolo/darknet/cfg/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load video
cap = cv2.VideoCapture(f'udp://192.168.10.1:11111?overrun_nonfatal=1&fifo_size=65536')

# Use the first frame to get the width and height
ret, test_frame = cap.read()
height, width, channels = test_frame.shape

# # Re-open the video file to start from the first frame
video_writer = cv2.VideoWriter(filename='output.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=20.0, frameSize=(width, height))

def detect_objects(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (64, 64), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs

def process_detections(frame, outs):
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return class_ids, confidences, boxes

def draw_boxes(frame, class_ids, confidences, boxes):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    if len(indexes) > 0:
        for i in indexes.flatten():
            if class_ids[i] == 0:  # Assuming '0' is the class ID for people
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame

# Main loop
while True:
    ret, frame = cap.read()
    if not ret: continue
    
    outs = detect_objects(frame)
    class_ids, confidences, boxes = process_detections(frame, outs)
    frame = draw_boxes(frame, class_ids, confidences, boxes)
    
    # if person_found:
    #     x, y, w, h = boxes[indexes.flatten()[0]]
    #     midscreen = width/2
    #     midbox = x + w/2
    #     #if box is to the left of the midpoint
    #     if (midbox < midscreen):
    #         if (midbox + w/2 < midscreen):
    #             print("rotate left")
    #         else:  
    #             print("centered")
    #     else :
    #         if (midbox - w/2 > midscreen):
    #             print("rotate right")
    #         else:  
    #             print("centered")
    # else:
    #     print('FIND PERSON')
    tello.rotate_clockwise(5)

    cv2.imshow("Frame", frame)
    video_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.land()
tello.streamoff()
cap.release()
video_writer.release()
cv2.destroyAllWindows()