import cv2
from yolo import yolo

print('TESTING YOLO MODEL')
capture = cv2.VideoCapture(1)
test_frame = capture.read()[1]
model = yolo(test_frame)
while True:
    print('Capturing')
    grabbed, frame = capture.read()
    if grabbed:
        class_ids, confidences, boxes = model.get_boxes(frame)
        frame = model.box_frame(frame, class_ids, confidences, boxes)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break