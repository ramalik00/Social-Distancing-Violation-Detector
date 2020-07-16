from detection import detect_people
import numpy as np
import argparse
import cv2
import imutils
import os


USE_GPU=False
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",type=str,default="",)
ap.add_argument("-o", "--output",type=str,default="")
ap.add_argument("-d", "--display",type=int,default=1,)
args = vars(ap.parse_args())


labelsPath=os.path.sep.join(["yolo", "coco.names"])
Labels=open(labelsPath).read().strip().split("\n")
weightsPath = os.path.sep.join(["yolo", "yolov3.weights"])
configPath = os.path.sep.join(["yolo", "yolov3.cfg"])
print("MODEL LOADED")
model=cv2.dnn.readNetFromDarknet(configPath,weightsPath)


if USE_GPU:
        
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


last_layer = model.getLayerNames()
last_layer=[last_layer[i[0] - 1] for i in model.getUnconnectedOutLayers()]
print("INPUT LOADED")
cap=cv2.VideoCapture(args["input"] if args["input"]!="" else 0)
writer = None

while True:
        
        (access, frame)=cap.read()
        if not access:
                break
        count=0
        
        frame = imutils.resize(frame, width=900)
        results = detect_people(frame,model,last_layer,Labels)
        a=set()
        if len(results)>= 2:

                centroids = np.array([r[3] for r in results])
                for i in range(0,len(centroids)):
                        for j in range(i+1,len(centroids)):
                                D=np.linalg.norm(centroids[i]-centroids[j])
                                if D <75:
                                        a.add(i)
                                        a.add(j)
        for (i,(prob,bounding_box,classes,centroid)) in enumerate(results):
                if i in a:
                        X_start, Y_start, X_end, Y_end = bounding_box
                        cv2.rectangle(frame,(X_start,Y_start),(X_end,Y_end),(0,0,255),2)
                        count+=1
                        cv2.putText(frame,str(count),((X_start+X_end)//2,(Y_start+Y_end)//2),cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,255,0),2)
                else:
                        X_start, Y_start,X_end,Y_end = bounding_box
                        cv2.rectangle(frame,(X_start,Y_start),(X_end,Y_end),(0,255,0),2)


                
        text = "Total Violations: "+str(len(a))
        cv2.putText(frame,text,(10, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 3)      
        if args["display"] > 0:
                
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                        break

        if args["output"] != "" and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 25,
                        (frame.shape[1], frame.shape[0]), True)

        if writer is not None:
                writer.write(frame)
