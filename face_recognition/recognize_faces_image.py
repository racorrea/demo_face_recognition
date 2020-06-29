#!/usr/bin/env python
# coding: utf-8


#Importamos los paquetes necesarios
import face_recognition
import argparse
import pickle
import cv2

#Construir el analizador de argumentos de la línea de comandos
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="Ruta al archivo que contiene las codificaciones faciales")
ap.add_argument("-i", "--image", required=True,
	help="Ruta a la imagen de prueba")
args = vars(ap.parse_args())

#Cargamos una fuente de texto:
font = cv2.FONT_HERSHEY_COMPLEX

#Cargamos los encodings conocidos
print("[INFO] cargando encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
#Cargamos la imagen de entrada y la convertimos de BGR a RGB
img = cv2.imread(args["image"])
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Detectamos las coordenadas (x, y)de los recuadros correspondientes
# a cada rostro en la imagen de entrada y luego calculamos los encodings de cada uno
print("[INFO] reconociendo rostros...")
loc_rostros = face_recognition.face_locations(rgb,
	model="cnn")
encodings_rostros = face_recognition.face_encodings(rgb, loc_rostros)
# Inicializamos el array de nombres
nombres_rostros = []

#Recorremos el array de encodings
for encoding in encodings_rostros:
	#Buscamos si hay alguna coincidencia con algún encoding conocido:
	coincidencias = face_recognition.compare_faces(data["encodings"],
		encoding)
	nombre = "Desconocido"
	#El array 'coincidencias' es ahora un array de booleanos.
    #Si contiene algun 'True', es que ha habido alguna coincidencia:
	if True in coincidencias:
		# Encuentra los índices de todas las caras coincidentes y luego
		# inicializa un diccionario para contar el número total de veces que cada 
		# rostro tuvo una coincidencia
		matchedIdxs = [i for (i, b) in enumerate(coincidencias) if b]
		counts = {}
		# Recorremos los índices coincidentes
		for i in matchedIdxs:
			nombre = data["names"][i]
			counts[nombre] = counts.get(nombre, 0) + 1
		# Determinamos el rostros con mayor número de coincidencias
		nombre = max(counts, key=counts.get)
	
	# Actualizamos la lista de nombres
	nombres_rostros.append(nombre)

#Dibujamos un recuadro rojo alrededor de los rostros desconocidos, y uno verde alrededor de los conocidos:
for ((top, right, bottom, left), nombre) in zip(loc_rostros, nombres_rostros):
	#Cambiar el color segun el nombre:
    if nombre != "Desconocido":
        color = (0,255,0) #Verde
    else:
        color = (0,0,255) #Rojo
 
    #Dibujar los recuadros alrededor del rostro:
    cv2.rectangle(img, (left, top), (right, bottom), color, 2)
    cv2.rectangle(img, (left, bottom - 20), (right, bottom), color, -1)
 
    #Escribir el nombre de la persona:
    cv2.putText(img, nombre, (left, bottom - 6), font, 0.6, (0,0,0), 1)

#Abrimos una ventana con el resultado:
cv2.imshow("Image", img)
print("\nMostrando resultado. Pulsa cualquier tecla para salir.\n")
cv2.waitKey(0)
cv2.destroyAllWindows()