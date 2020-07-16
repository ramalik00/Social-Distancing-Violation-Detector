# Social Distancing Violation Detector
# MODEL
###  The Detector uses YOLOv3 model trained on COCO Dataset to detect people.In case you want to train the model on your own you need to have Darknet in your system.Darknet is    framework to train neural networks, it is open source and written in C/CUDA and serves as the basis for YOLO.
###  It is fast, easy to install, and supports CPU and GPU computation
# FUNCTIONING
### The model first detects persons in the frame and computes distance between all the pairs . If the minimum distance is less than 6ft then that pair is shown in red color ,otherwise in green. The important thing to do before is calibrate the camera otherwise it will give incorrect results . Link to calibrate camera using opencv https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
###  Keep all the downloaded files in same working directory
###  Execute main.py from terminal or command line
###  Use following command line arguments:
     --input:to input your image or video file to detect objects in it  
     --output:to save your output file after object detection by the model with same same format as of input.Prefer video file is saved as .avi

     --display 0:if you don't want to see object detection in your input in real time
