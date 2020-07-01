#!/usr/bin/env python
# coding: utf-8

import os
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ## Captura de imágenes para entrenamiento
#  
# En el siguiente código la variable **id** representa el identificador por cada persona con la que se desea entrenar el modelo, este debe incrementar por cada nuevo rostro. 
# 
# Los pasos siguientes se enciende la cámara del dispositivo para obtener los rostros con ayuda del modelo para la detección de los mismo, que se encuentra en la variable: **face_cascade**, para mejorar la precisión la imagen se convierte a escala de grises y se configuran los parámetros para la detección de rostros en **1.1**, este factor controla el reescalado de la imagen que es de gran importancia para detectar rostros según su tamaño en la imagen.
# 
# En la carpeta **img_train** se almacenan los rostros obtenidos que en este ejemplo se toman un total de 20, el nombre de cada archivo contiene el formato **User.1.1** el primer valor es el de la variable **id** y el siguiente un incremental por las 20 imágenes que se van a generar.


def getImages(id, source):
	print("INFO Obteniendo rostros de usuario " + str(id) + "...")
	cap = cv2.VideoCapture(source)

	cont=0; #Contador por cada rostro detectado y almacenado

	facesLimit=30; #Cantidad de rostros a obtener

	while 1:

		ret, img = cap.read() #Inicia cámara y obtiene fotogramas
		
		if (ret):

			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convierte la imagen a escala de grises

			faces = face_cascade.detectMultiScale(gray, 1.2, 5)

			for (x,y,w,h) in faces:

				cont=cont+1;

				cv2.imwrite("img_train/User."+str(id)+ "." +str(cont)+ ".jpg", gray[y:y+h, x:x+w])

				cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

				cv2.waitKey(100)

			cv2.imshow('img',img)

			cv2.waitKey(1)

			if cont > facesLimit:

				break
		else:
			break

	cap.release()
	cv2.destroyAllWindows()


# Para obtener imágenes desde la cámara se debe especificar el valor **id** (primer parámetro) y el valor **source** (segundo parámetro), para el cual especificar **0** corresponde a usar la cámara mientras que si se especifica una ruta, se tomara las capturas desde un video.

#getImages(1, 0) #Obtener imagenes desde camara
getImages(1, 'video_train/train_1.mp4') #Obtener imagenes desde video id:1
getImages(2, 'video_train/train_2.mp4') #Obtener imagenes desde video id:2
