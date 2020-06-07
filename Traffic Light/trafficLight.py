from imageai.Detection import ObjectDetection
import os
execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "3.jpeg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))
from PIL import Image
img = Image.open("imagenew.jpg")
area = detections[0]['box_points']
cropped_img = img.crop(area)
cropped_img.save('imagenew.jpg')

import cv2
import numpy as np
img = cv2.imread('imagenew.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
green = 60;
yellow = 30
#green
lower_range = np.array([60 - 20, 100, 100]) 
upper_range = np.array([60 + 20, 255, 255])
mask = cv2.inRange(hsv, lower_range, upper_range)
unique, counts = np.unique(mask, return_counts=True)
g=dict(zip(unique, counts))
#yellowlower_range = np.array([60 - 15, 100, 100]) 
lower_range = np.array([30 - 15, 100, 100]) 
upper_range = np.array([30 + 15, 255, 255])
mask_0 = cv2.inRange(hsv, lower_range, upper_range)
unique, counts = np.unique(mask_0, return_counts=True)
y=dict(zip(unique, counts))
#red
lower_range = np.array([169, 100, 100], dtype=np.uint8)
upper_range = np.array([189, 255, 255], dtype=np.uint8)
mask_1 = cv2.inRange(hsv, lower_range,upper_range )
unique, counts = np.unique(mask_1, return_counts=True)
r=dict(zip(unique, counts))
max1=0
if(len(g)>1):
    max1=g[255]
    color='green'
if(len(y)>1):
    if(max1<y[255]):
        max1=y[255]
        color='yellow'
if(len(r)>1):
    if(max1<r[255]):
        max1=r[255]
        color='red'
        
print(color)
