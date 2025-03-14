# -*- coding: utf-8 -*-
"""TP2Photos.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rQ0jBon31ZCVx2G9FjYg9-5a-LXMscsP
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.utils import class_weight as cw
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import Recall
from keras import backend as K

from google.colab import drive
drive.mount('/content/drive')

# Chemin vers ton dossier d'images dans Google Drive
image_folder = '/content/drive/My Drive/fichierTPphotos/training_set/cats/'
images = os.listdir(image_folder)

for i in range(0, len(images)) :
  images[i] = image_folder + images[i]

# Définir ImageDataGenerator
#
# La fonction ImageDataGenerator sert à adapter l'image pour qu'elle puisse
# ensuite être
# traitée par le réseau de neurones
# Le paramètre rescale sert à normaliser la valeur de chaque pixel entre 0 et 1,
#  ça permet d'entraîner le modèle plus rapidement.

train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)
train_dataset = train.flow_from_directory("/content/drive/My Drive/fichierTPphotos/training_set/", target_size=(150,150), batch_size = 32, class_mode = 'binary')
test_dataset = test.flow_from_directory("/content/drive/My Drive/fichierTPphotos/training_set/", target_size=(150,150), batch_size =32, class_mode = 'binary')

# - Expliquer le réseau de neurones ci-dessous :
#
# Ce réseau de neurones est composé de 7 couches. Ce modèle est conçu pour une
# classification binaire
# Première commade : Sequential() sert à empiler les couches les unes sur les
# autres linéairement.
#
# La fonction Conv2D(nbFiltres, (tailleFiltre, tailleFiltre),
# activation = 'relu',
# input_shape=(largeurPixelImageEntrée, hauteurPixelImageEntrée, nbCanaux))
# sert à effectuer une convolution avec 'nbFiltres' filtres de taille
# (tailleFiltre x tailleFiltre). pour détecter les caractéristiques locales des
# images
# le reste des paramètres sert à supprimer les valeurs négatives obtenues
# (activation = 'relu') pour les mettre à la valeur 0, puis à définir les images
# en entrée
# La fonction MaxPool2D(tailleFenetre, tailleFenetre) est une couche de pooling
# qui prend le maximum de la fenêtre, pour simplifier le résultat
# et les prochains traitements dans les autres couches.
# Cette fonction réduit la taille de la matrice et conserve les informations
# importantes de l'image.
#
# A chaque nouvelle couche convolutive, grâce au nombre de filtres,
# on essaye d'extraire/ d'identifier encore plus de
# caractéristiques locales de l'images
# On obtient à la fin des quatre couches convolutives, les motifs
# complexes et difficiles à trouver de l'image.
#
# La fonction Flatten() sert à réduire le nombre de paramètres avant que
# le résultat soit envoyé aux couches fully-connected
#
# La fonction Dense(nbNeurones, activation='relu') est la
# couche fully-connected du modèle, elle intègre 'nbNeurones' neurones
#
# La dernière couche fully-connected sert à classifier de manière binaire le
# résultat car son nombre de neurones est de 1.
# Tandis que le paramètre : activation='sigmoid' permet d'avoir un résultat
# entre 0 et 1.

model = tf.keras.Sequential()

# Convolutional layer and maxpool layer 1
model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(tf.keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 2
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 3
model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 4
model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))

# This layer flattens the resulting image array to 1D array
model.add(tf.keras.layers.Flatten())

# Hidden layer with 512 neurons and Rectified Linear Unit activation function
model.add(tf.keras.layers.Dense(512,activation='relu'))

# Output layer with single neuron which gives 0 for Cat or 1 for Dog
# Here we use sigmoid activation function which makes our model output to lie between 0 and 1
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

# Expliquer les paramètres (optimizer, loss, metrics)

#La fonction compile permet de determiner le processus d'apprentissage, ici
# ce serait l'algorithme d'optimisation 'adam' grâce au paramètre optimizer
# Le paramètre loss permet de mesurer la difference entre la prédiction du
# modèle et l'étiquette de l'image
# Le paramètre metrics sert à mesurer la performance du modèle ou plutôt selon
# Keras 'à juger la performance du modèle'

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy', Recall(), f1])

model.fit(
        train_dataset,
        steps_per_epoch=1000,
        epochs=10,
        validation_data=test_dataset,
        validation_steps=500
        )

from tensorflow.keras.preprocessing import image
import numpy as np
img = image.load_img(images[2], target_size=(150, 150))  # Même taille que celle de l'entrainement
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0
prediction = model.predict(img_array)
print(prediction)

if prediction[0] > 0.5:
    print(f"Il n'y a probablement pas de chat (probabilité : {prediction[0][0]:.2f})")
else:
    print(f"Il y a probablement un chat (probabilité : {prediction[0][0]:.2f})")
