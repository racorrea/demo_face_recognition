#!/usr/bin/env python
# coding: utf-8


# Importar paquetes necesarios
from imutils import paths
import face_recognition  #Librería 
import argparse
import pickle
import cv2
import os

#Declaramos las variables de entrada
datasetPath = "dataset"
encodingFile = "encodings.pickle"

# Almacenar las rutas a las imágenes de entrada en nuestro conjunto de datos
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(datasetPath))
# Inicializar la lista de codificaciones conocidos y nombres conocidos
#Los encodings son las características únicas de cada rostro que permiten diferenciarlo de otros.
knownEncodings = []
knownNames = []


#Recorremos las rutas de las imagenes
for (i, imagePath) in enumerate(imagePaths):
	#Extreamos el nombre de la persona de la ruta de la imagen 
	#(nombre de la carpeta que almacena clos rostros de cada persona)
	print("[INFO] Procesando imagen... {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	# Cargamos la imagen y convertimos de BGR (OpenCV ordering)
	# a RGB (dlib ordering)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# Detectamos las coordenadas(x, y) de los recuadros
	# correspondiente a cada rostro en la imagen de entrada
	boxes = face_recognition.face_locations(rgb,
		model="cnn")
	#Calculamos los encondings del rostro
	encodings = face_recognition.face_encodings(rgb, boxes)
	#Recorremos el array de encodings que hemos encontrado
	for encoding in encodings:
		#Añadimos cada codificación + nombre a nuestro set de nombres conocidos y
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

#Almacenamos los encodings + nombres en el archivo especificado
print("[INFO] Serializando encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(encodingFile, "wb")
f.write(pickle.dumps(data))
f.close()