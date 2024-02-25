# YOLO (Object Detection)

In order for the current scripts to work properly, the external files to be downloaded are anticipated in this directory:

- `/yolo/darknet`
    - Source: [DarkNet](https://pjreddie.com/darknet/yolo/)
    - `git clone https://github.com/pjreddie/darknet`
        - Clones the DarkNet repository
- `/yolo/yolov3.weights`
    - `wget https://pjreddie.com/media/files/yolov3.weights` (Windows)
    - `curl -O https://pjreddie.com/media/files/yolov3.weights` (Mac/Linux)
        - Weights for the YOLO model