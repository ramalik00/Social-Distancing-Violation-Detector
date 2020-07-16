# Social-Distancing-Violation-Detector



#### Install Tensorflow,Imutils,OpenCV(Manual Installation for GPU access otherwise use pip if you want to do it on CPU)

#### Download YOLOv3-416 weights file from here: https://pjreddie.com/darknet/yolo/

#### Now make a folder with the name yolo in the working directory and place the above weights file along with yolov3.cfg and coco.names files in this folder

#### Make sure you use the above folder name only otherwise you would have to change the path to model in both object_detection.py and object_detection_image.py

#### You can also use YOLOv4 but the latest version of opencv till now does not support it , So in order to use YOLOv4 , you will have to manually build Opencv 3.4.0 .
#### Tutorial for manual build https://github.com/opencv/opencv/pull/17185

