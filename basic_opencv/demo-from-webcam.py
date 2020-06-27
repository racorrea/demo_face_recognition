import cv2 as cv

# Get user supplied values
cascPath = "haarcascade_frontalface_default.xml"
# Create the haar cascade
faceCascade = cv.CascadeClassifier(cascPath)


#variables

# Start coordinate, here (5, 5) 
# represents the top left corner of rectangle 
start_point = (5, 5) 
  
# Ending coordinate, here (220, 220) 
# represents the bottom right corner of rectangle 
end_point = (220, 220) 
  
# Blue color in BGR 
color = (139, 242, 28) 
  
# Line thickness of 2 px 
thickness = 2
  
# Using cv2.rectangle() method 
# Draw a rectangle with blue line borders of thickness of 2 px 
#image = cv2.rectangle(image, start_point, end_point, color, thickness) 

#finvariables



webcamCapture = cv.VideoCapture(0)

if not webcamCapture.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Captura el video frame por frame
    ret, frame = webcamCapture.read()


    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    imagen_gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #Coordenadas rostros
    coordenadas_rostros = faceCascade.detectMultiScale(imagen_gris, 1.3, 5)

    #recorrer dats
    for (x,y,ancho,alto) in coordenadas_rostros:
        cv.rectangle(frame, (x,y), (x+ancho, y+alto), color, thickness) 

    cv.imshow('Output', frame)

    # Display the resulting frame
    #cv.imshow('frame', imagen_gris)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
webcamCapture.release()
#cv.destroyAllWindows()