#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import cv2
import os
from PIL import Image # For face recognition we will the the LBPH Face Recognizer 


# ## Prueba modelo en tiempo real con la cámara del dispositivo

# In[173]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("face_model/trainingdata.yml")
id=0

fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255,0,0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 3)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w]) 
        if(id==1):
            id="Keanu" #definir nombre de usuario 1
        if id==2:
            id="Laurence" #definir nombre de usuario 2
        cv2.putText(img, str(id), (x,y+h), fontface, fontscale, fontcolor) 
        
    cv2.imshow('img',img)
    
    #Salir con 'ESC':
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        break
        
cap.release()

cv2.destroyAllWindows()