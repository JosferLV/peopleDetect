import numpy as np
import cv2
import imutils
from imutils.object_detection import non_max_suppression
from matplotlib import pyplot as plt

cap=cv2.VideoCapture(0) #Modificamos el código para que tome la imagen de la camára web
hog = cv2.HOGDescriptor()#Creamos el objeto HOG
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())#Utilizamos el detector de personas por defecto del objeto HOG
fourcc = cv2.VideoWriter_fourcc(*'XVID')#Establecemos el formato de compresión del video 
out = cv2.VideoWriter('videoCamara.avi',fourcc, 40.0, (640,480))#Guardamos el video que se genere 

while True:
    ret, imagen = cap.read()#Lee la imagen 
    (rectas,weights)=hog.detectMultiScale(imagen,winStride=(8,8),padding=(5,5),scale=1.05)#Escala de la imagen 
    rectas=np.array([[x,y,x+w,y+h] for (x,y,w,h) in rectas])#Establece las cordenadas en donde encuentre 
    eleccion=non_max_suppression(rectas,probs=None,overlapThresh=0.65)#Elegimos el area en donde es mas probable que se encuentren las coincidencias
    for (xA,yA,xB,yB) in eleccion:
        cv2.rectangle(imagen,(xA,yA),(xB,yB),(0,255,0),2) #Pinta las rectas en la deteccion
    out.write(imagen)#escribe
    cv2.imshow("imagen de salida",imagen)#Muestra la pantalla
    if(cv2.waitKey(1) & 0xFF == ord('q')):#Key para salir 
        break
cap.release()
out.release()
cv2.destroyAllWindows()