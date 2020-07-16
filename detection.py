import numpy as np
import cv2
from imutils.object_detection import non_max_suppression

Min_Conf=0.3
def detect_people(frame,net,ln,Labels):
	(H,W)=frame.shape[:2]
	results=[]
	blob=cv2.dnn.blobFromImage(frame,1/255.0,(416, 416),swapRB=True,crop=False)
	net.setInput(blob)
	layerOutputs=net.forward(ln)
	bounding_box=[]
	probabilities=[]
	classes=[]
	centroids=[]

	for output in layerOutputs:
		for detection in output:
			scores=detection[5:]
			classID=np.argmax(scores)
			probability=scores[classID]
			if probability>Min_Conf:
				if Labels[classID]=="person":
                                        
                                        box = detection[0:4]*np.array([W, H, W, H])
                                        (X_center,Y_center,w,h)=box.astype("int")
                                        X_start=int(X_center-(w/2))
                                        Y_start=int(Y_center-(h/2))
                                        
                                        bounding_box.append([X_start,Y_start,X_start+int(w),Y_start+int(h)])
                                        centroids.append((X_center,Y_center))
                                        probabilities.append(float(probability))
                                        classes.append(classID)
				



				

	indexs =non_max_suppression(np.array(bounding_box),probs=probabilities)

	
	if len(indexs)>0:
		
		for i in range(len(indexs)):
			(X_start,Y_start,X_end,Y_end)=(bounding_box[i][0],bounding_box[i][1],bounding_box[i][2],bounding_box[i][3])
			r=(probabilities[i],(X_start,Y_start,X_end,Y_end),classes[i],centroids[i])
			results.append(r)

	
	return results
