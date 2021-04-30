# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 23:12:59 2021

@author: david
"""
#Estas librerias sirven para poder movernos entre carpetas dentro del sistema operativo
import sys
import os
import matplotlib.pyplot as plt
#Esta función va a yudarnos a preprocesar las imagenes que vamos a dar al algoritmo
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
#El optimizador con el que vamos a entrenar al modelo
from tensorflow.python.keras import optimizers
#Nos permite crear redes neuronales secuenciales, es decir, que cada una de las capas está en orden
from tensorflow.python.keras.models import Sequential
#
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
#Las capas donde vamos a hacer las convoluciones
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
#Esto es implrtante, porque si hay algna función de keras en el backgroung
#con esta funcón las eliminamos y trabajamos con la cpu libre
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import applications

#Limpiamos los procesos de keras que puedieran estar ejecutandose en el background
K.clear_session()

data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

#Numero de veces que vamos a iterar sobre todo el set de entrenamiento
epocas=20
#Ajustamo la altura y la longitud a 150 pixeles
longitud, altura = 28, 28
#Numero de imagenes que mando a procesar en cada uno de los pasos
batch_size = 32
#Una epoca va a tener 1000 pasos
pasos = 1200
#300 pasos para la validación en cada época
validation_steps = 300
#Numero de filtros para la convolución.
filtrosConv1 = 32#Profundidad de 32
filtrosConv2 = 64#Profundidad de 64
#Definimos el tamaño del filtro
tamano_filtro1 = (2, 2)#Para la primera convolución
tamano_filtro2 = (2, 2)#Para la segunda
tamano_pool = (2, 2)#Tamaño del filtro en el max pulling
#tenemos 10 clases. 10 
clases = 10
#Error en el apredizaje //learning rate
lr = 0.0005

"""
Pre procesamiento de nuestras imagenes
Rescalamos las imagenes de 0 a 255 pixeles
shear_range->inclina un poco algunas imagenes
zoom_range->acerca un poco a algunas imagenes
horizontal_flip->invertirá algunas imagenes
"""
entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

"""
categorical->Vamos a hacer una clasificación categorical, es decir, 
tenemos que clasdificas las imagenes en clases
"""
entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')


 
#Creo la red neuronal
#Indicamos que se trata de una red secuencial
cnn = Sequential()

"""
Creamos a primera capa, esta se trata de una convolución.
Definimos la cantidad de filtros que conforman la capay su tamaño


padding=same-> Lo que va a hacer el filtro cuando llegue a las esquinas
input_shape-> Las imagenes que vamos a enviar a la primera capa
tienen una altura la cual ya hemos definido antes 
Relu-> función de activación

MaxPooling->capa de maxpooling
"""
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
"""
Lo mismo que la capa anterior, pero con otros filtros.
No hace falta definir otra vez el tamaño de las imagenes
"""
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
"""
Flatten->La imagenq que hemos obtenido que es muy profunda y muy pequeña. la aplanamos
Dense-> Después de aplanar la imagen, mandamos la imagen a una red con 256 neuronas y la función de activación es relu
Es como una capa normal
Dropout->Con esto lo que etsaos haciendo es inabilitar el 50% de forma aleatoria de las neuronas en cada paso
Hacemos esto para evitar sobreajustamientos. Es decir, si están activadas todas la neuronas
puede suceder que la red aprenda a escoger siempre el mismo camino para clasificar un tipo de foto.
De esta forma forzamos a que el modelo desarrolle varios caminos.
Lorgrando así que el modelo sea más versatil y capz con información nueva.
Dense->Última capa formada por 10 neuronas (clases) con activación de softmax.
Esta activación lo que hace es devolver un tanto porciento deporibilidades
de que puede ser la imagen.
Ejemplo:
    10%->1
    5%->2
    1%->3
    40%->4
    .
    .
    .

"""
cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))

"""
La función de perdida sirve para que durante el entrenamiento, la red neuronal sepa que tan bien o que tan
mal lo está haciendo. Es como una retroalimentación de como lo está haciendo.

Definimos que es de tipo categorical, por que separamos en categorical
Usamos el optimizador adam
La métrica indicará en porcentaje que tan bien o mal está aprediendo la red
"""
cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=cnn.fit(
        entrenamiento_generador,
        steps_per_epoch=pasos,
        epochs=epocas,
        validation_data=validacion_generador,
        validation_steps=validation_steps)

target_dir = './modelo/'
#Si no hay una carmeta llamada modelo la creas
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
  
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()