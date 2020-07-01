#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import cv2
import os
from PIL import Image # For face recognition we will the the LBPH Face Recognizer 

# ## Prueba modelo con una imagen

# In[174]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("face_model/trainingdata.yml")
id=0

fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255,0,0)

#Se define la imagen de prueba
img = cv2.imread('img_test/1.jpg')

scale_percent = 100 # percent of original size

width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)

dim = (width, height)
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.5, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    id,conf=rec.predict(gray[y:y+h,x:x+w])
    if id==1:
        id="Keanu" #definir nombre de usuario 1
    if id==2:
        id="Laurence" #definir nombre de usuario 2
    cv2.putText(img, str(id), (x,y+h), fontface, fontscale, fontcolor) 
    
cv2.imshow('img',img)
print("\nMostrando resultado. Pulsa cualquier tecla para salir.\n")
cv2.waitKey(0)
cv2.destroyAllWindows()