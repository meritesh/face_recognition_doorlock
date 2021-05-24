import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
# image recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# loading the training set images
imgpath="dataset"
image_path=[]
# saving all the image paths in list image_path
for i in (os.listdir(imgpath)):
    image_path.append(os.path.join(imgpath,i))


# kind of an anomaly here since the cv2 method and PIL are producing different numpy array for supposedly same image.-----

# print(image_path[1])
# print(len(image_path))
# imagesss=cv2.imread(image_path[0]) 
# print(imagesss.shape)
# cv2.imshow('dd',imagesss)
# cv2.waitKey(0)   

# CREATING THE LABELS FOR IMAGES AND LOADING IMAGES AS NUMPY ARRAYS FOR THE RECOGNIZER
faces=[]
labels=[]
for ima in image_path:
    # using PIL to extract image at the specified image address in greyscale format
    faceImg=Image.open(ima).convert('L')
    print(faceImg)
    # converting the image into numpy array
    faceNp=np.array(faceImg,'uint8')
    faces.append(faceNp)
    # extracting the label of the image
    L=int(os.path.split(ima)[-1].split('.')[1])
    labels.append(L)
    print(L)
    # cv2.imshow('dd',faceNp)
    # cv2.waitKey(0) 
    
print(faceNp.shape)

# training and saving the produced face recognizer
recognizer.train(faces,np.array(labels))
recognizer.save('recognizer_training.yml')
