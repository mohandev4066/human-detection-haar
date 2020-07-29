import numpy as np
import os.path
import cv2
# Create our body classifier
body_classifier = cv2.CascadeClassifier(os.path.join(cv2.haarcascades, 'haarcascade_upperbody.xml'))
# Initiate video capture for video file
cap = cv2.VideoCapture(0)
dummy_array=np.array([])

cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

# Loop once video is successfully loaded
while cap.isOpened():
    ret, frame = cap.read()
    frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray ,1.01, 3)
    
    if type(bodies) == type(dummy_array):
    	for (x,y,w,h) in bodies:
    		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
    		cv2.putText(frame, "occupied", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
    		cv2.imshow("frame",frame)
    	
    else:	
    	cv2.putText(frame, "empty", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
    	cv2.imshow("frame",frame)
    		

    if cv2.waitKey(0):
    	break

cap.release()
cv2.destroyAllWindows()
