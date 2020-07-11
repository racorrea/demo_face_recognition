#!/usr/bin/env python
# coding: utf-8


import numpy as np #Trabajar con matrices
import cv2 #OpenCv
import os
from PIL import Image # For face recognition we will the the LBPH Face Recognizer 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') #Cargan el algoritmo Haarcascade Classifier


#  ## Creación de modelo
#  
# El siguiente bloque contiene el entrenamiento del modelo para el reconocimiento facial, se toman las imágenes generadas en la carpeta **img_train** por cada persona y obtiene las características encontradas relacionándolas al **id** de cada imagen.

# In[172]:


recognizer = cv2.face.LBPHFaceRecognizer_create() #Se crea instancia al modelo LBPH para el reconocimiento facial

path="img_train"

def getImagesWithID(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]   # Obtiene imagenes del directorio

    faces = []

    IDs = []

    for imagePath in imagePaths: # Loop por cada imagen encontrada

        facesImg = Image.open(imagePath).convert('L') # Convierte la imagen a escala de grises

        faceNP = np.array(facesImg, 'uint8') # Genera matriz con pixeles de la imagen 

        ID= int(os.path.split(imagePath)[-1].split(".")[1]) #Obtiene id asignnado a la imagen

        faces.append(faceNP) # Asocia características de los rostros

        IDs.append(ID) # Almacena id por conjunto de imagenes analizadas

        cv2.imshow("Rostros para entrenamiento",faceNP) # Muestra en pantalla el proceso

        cv2.waitKey(10)

    return np.array(IDs), faces # Devuelve matrices por cada imagen asociada a una misma persona y el respectivo id


Ids,faces  = getImagesWithID(path) # Llama a función para obtener matrices de las imagenes

recognizer.train(faces,Ids) # Entrena el modelo

recognizer.save("face_model/trainingdata.yml") # Se almacena el modelo

cv2.destroyAllWindows() # Se cierran las ventanas abiertas

