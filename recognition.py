import cv2
import numpy as np

# opening camera
cap=cv2.VideoCapture(0)
# for detecting the face
face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

rec=cv2.face.LBPHFaceRecognizer_create()
# declaring the self-made trained recognizer
rec.read('recognizer_training.yml')

#using Hershey font to write in the image.
font = cv2.FONT_HERSHEY_SIMPLEX

# checking if the camera is opened properly.
if(cap.isOpened()):
    ret,frame=cap.read()
else:
    ret=False

while(ret):
    _,frame=cap.read()
    g_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # detecting the face
    faces=face_detector.detectMultiScale(g_frame,1.05,4)
    # if face found
    if(len(faces)>0):
        for(x,y,w,h) in faces:
            # running the face array in recognizer and getting the labels as output.
            id,conf=rec.predict(g_frame[y:y+h,x:x+h])
            # adding the label in the image itself.
            if((id==1 or id==2 or id==3) and conf<36):
                cv2.putText(frame, str(id), (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
                print(conf,id)
            # printing on the console as well.
            
    else:
        print("4")
    cv2.imshow('hehe',frame)
    if(cv2.waitKey(1)& 0xFF==ord('b')):
        break
    
# closing the camera and any open window on the screen.
cap.release()
cv2.destroyAllWindows()



