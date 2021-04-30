# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:29:04 2021

@author: david
"""
import numpy as np
import cv2
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud, altura=28,28
modelo='./modelo/modelo.h5'
pesos='./modelo/pesos.h5'
cnn=load_model(modelo)
cnn.load_weights(pesos)


def predict(file):
    x=load_img(file,target_size=(longitud,altura))
    x=img_to_array(x)
    x=np.expand_dims(x,axis=0)
    arreglo=cnn.predict(x) 
    resultado=arreglo[0]
    respuesta= np.argmax(resultado)
    if respuesta==0:
        print('Numero 0')
    elif respuesta==1:
        print('Numero 1')
    elif respuesta==2:
        print('Numero 2')
    elif respuesta==3:
        print('Numero 3')
    elif respuesta==4:
        print('Numero 4')
    elif respuesta==5:
        print('Numero 5')
    elif respuesta==6:
        print('Numero 6')
    elif respuesta==7:
        print('Numero 7')
    elif respuesta==8:
        print('Numero 8')
    elif respuesta==9:
        print('Numero 9')
        
    return respuesta

img='./25.png'
imgcv2=cv2.imread(img)
cv2.imshow(' ',imgcv2)
cv2.waitKey(0)
predict(img)
    
