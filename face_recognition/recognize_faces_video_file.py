#!/usr/bin/env python
# coding: utf-8


#Importamos los paquetes necesarios
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

#Ruta al archivo que contiene las codificaciones faciales
encodingFile = "encodings.pickle"

#Construir el analizador de argumentos de la línea de comandos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="Ruta al video de entrada")
ap.add_argument("-o", "--output", type=str,
	help="Ruta al video de salida")
ap.add_argument("-y", "--display", type=int, default=1,
	help="Si se muestra o no el Video de salida en la pantalla")
args = vars(ap.parse_args())

#Cargamos una fuente de texto:
font = cv2.FONT_HERSHEY_COMPLEX

#Cargamos los encodings conocidos
print("[INFO] Cargando encodings...")
data = pickle.loads(open(encodingFile, "rb").read())

#Inicializamos el puntero al archivo de video y al escritor de video
print("[INFO] procesando video...")
stream = cv2.VideoCapture(args["input"])
writer = None

# Recorremos los frames del video de entrada
while True:
	#Leemos el siguiente frame
	(grabbed, frame) = stream.read()

	#Si el siguiente frame no fue leído, entonces se ha llegado
	#al final del video
	if not grabbed:
		break

	# Convertimos el frame de BGR to RGB luego modificamos el tamaño
	# a 750px (para acelarar el procesamiento)
	#rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])

	# Detectamos las coordenadas (x, y)de los recuadros correspondientes
	# a cada rostro en la imagen de entrada y luego calculamos los encodings de cada uno
	loc_rostros = face_recognition.face_locations(rgb,
		model="hog")
	encodings_rostros = face_recognition.face_encodings(rgb, loc_rostros)
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
		#Deshacemos la reducción de tamaño para tener las coordenadas de la imagene original 
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom *r)
		left = int (left * r)
		
		#Cambiar el color segun el nombre:
		if nombre != "Desconocido":
			color = (0,255,0) #Verde
		else:
			color = (0,0,255) #Rojo
	 
		#Dibujar los recuadros alrededor del rostro:
		cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
		cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, -1)
	 
		#Escribir el nombre de la persona:
		cv2.putText(frame, nombre, (left, bottom - 6), font, 0.5, (0,0,0), 1)

	# Si el escritor del video es NONE, inicializamos el writer. 
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		writer = cv2.VideoWriter(args["output"], fourcc, 20.0,
			(frame.shape[1], frame.shape[0]),True)

	# Si el escritor no es NONE, escribimos el frame con los rostros reconocidos
	if writer is not None:
		writer.write(frame)

	# Verificamos si debemos presentar en pantalla el video de salida o no.
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# Si se presiona la letra "q", se termina el loop.
		if key == ord("q"):
			break

# Cerrar los punteros
cv2.destroyAllWindows()
stream.release()

# Vericamos si es necesario liberar el puntero del escritor de video
if writer is not None:
	writer.release()
