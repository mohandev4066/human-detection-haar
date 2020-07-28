import numpy as np
import os.path
import cv2
# Create our body classifier
body_classifier = cv2.CascadeClassifier(os.path.join(cv2.haarcascades, 'haarcascade_upperbody.xml'))
# Initiate video capture for video file
cap = cv2.VideoCapture(0)
# Loop once video is successfully loaded
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray ,1.1, 3)
    
    
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Pedestrians', frame)
    if cv2.waitKey(1) == ord("q"):
    	break
cap.release()
cv2.destroyAllWindows()