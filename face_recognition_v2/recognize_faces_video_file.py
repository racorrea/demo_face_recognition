#!/usr/bin/env python
# coding: utf-8

# # Detección de rostros y reconocimiento facial
# 
# ## Librerías
# 
# Este proyecto está desarrollado bajo la **versión 3.8 de python** por medio del entorno de **Anaconda para Windows**, además se hará uso de la última versión de **openCV** en conjunto con el algoritmo **LPBH** para la creación de un modelo para el reconocimiento facial.
# 
# > **Anaconda:** https://docs.anaconda.com/anaconda/install/windows/
# 
# > **OpenCV:** conda install -c conda-forge opencv
# 
# > **Modelo Detección de rostros:** Descargar archivo "haarcascade_frontalface_alt.xml" desde https://github.com/opencv/opencv/tree/master/data/haarcascades

# In[169]:


import numpy as np 
import cv2
import os
from PIL import Image # For face recognition we will the the LBPH Face Recognizer 

# ## Prueba modelo con un video

# In[175]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture('video_test/1.mp4')
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("face_model/trainingdata.yml")
id=0

fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255,0,0)

while cap.isOpened():
    ret, img = cap.read()
    if ret == True:
        scale_percent = 50 # percent of original size

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
                id="Keanu " #definir nombre de usuario 1
            if id==2:
                id="Laurence " #definir nombre de usuario 2
            cv2.putText(img, str(id), (x,y+h), fontface, fontscale, fontcolor) 

        cv2.imshow('img',img)

        #Salir con 'ESC':
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break
    else:
        break
        
cap.release()

cv2.destroyAllWindows()

