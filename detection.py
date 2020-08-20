import numpy as np
import cv2
from imutils.object_detection import non_max_suppression

#Getting Bird's Eye perspective for the frame
def compute_perspective_transform(corner_points,width,height,img):
       
        corner_points_array = np.float32(corner_points)
        img_params = np.float32([[0,0],[width-1,0],[0,height-1],[width-1,height-1]])
        matrix = cv2.getPerspectiveTransform(corner_points_array,img_params) 
        return matrix

def compute_point_perspective_transformation(matrix,centroids):
        if len(centroids)==0: 
                return (centroids)
        
        
        list_points_to_detect=np.float32(centroids).reshape(-1, 1, 2)
        transformed_points=cv2.perspectiveTransform(list_points_to_detect, matrix)
       
        transformed_centroids=[]
        for i in range(0,transformed_points.shape[0]):
                transformed_centroids.append([transformed_points[i][0][0],transformed_points[i][0][1]])
        return transformed_centroids

Min_Conf=0.3
def detect_people(frame,model,last_layer,Labels):
	(H,W)=frame.shape[:2]
	#Enter the corner points of the contour for getting bird's eye perspective after callibrating the camera
	r1,c1=(,)
        r2,c2=(,)
        r3,c3=(,)
        r4,c4=(,)

        matrix=compute_perspective_transform(((r3,c3),(r2,c2),(r4,c4),(r1,c1)),W,H,frame);
	
	results=[]
	blob_image=cv2.dnn.blobFromImage(frame,1/255.0,(416, 416),swapRB=True,crop=False)
	model.setInput(blob_image)
	layerOutputs=model.forward(last_layer)
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
                                        centroids.append((X_center,Y_start))
					centroids=compute_point_perspective_transformation(matrix,centroids)
                                        probabilities.append(float(probability))
                                        classes.append(classID)
				



				

	indexs =non_max_suppression(np.array(bounding_box),probs=probabilities)

	
	if len(indexs)>0:
		
		for i in range(len(indexs)):
			(X_start,Y_start,X_end,Y_end)=(bounding_box[i][0],bounding_box[i][1],bounding_box[i][2],bounding_box[i][3])
			r=(probabilities[i],(X_start,Y_start,X_end,Y_end),classes[i],centroids[i])
			results.append(r)

	
	return results
