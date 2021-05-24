import cv2
outpath="dataset"
# haar cascade classifier files from opencv github
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')

cap=cv2.VideoCapture(0)
count=0
if (cap.isOpened()):
    ret,frame=cap.read()
else:
    ret=False

while(ret):
    ret,frame=cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detecting faces
    faces=face_cascade.detectMultiScale(frame_gray,1.05,4)
    print(faces)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
        # if the face found then saving it along with the label
        cv2.imwrite(outpath +"\\frame.1.%d.jpg" % count,faceROI)
        count+=1
        #-- In each face, detect eyes
        eyes = eye_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
        
    cv2.imshow('trial',frame)
    if(cv2.waitKey(1) & 0xFF== ord('b')):
        break
    
cv2.destroyAllWindows()
cap.release()    
    
    
    
