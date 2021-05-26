import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

data = np.load('face_images.npz')['face_images']
data = np.swapaxes(np.swapaxes(data, 1, 2), 0, 1)
data = data

label = np.array(pd.read_csv('facial_keypoints.csv'))
eye = label[:, :4]
mouth = label[:, 28:]

a = []
for i in range(len(data[:,0])):
    if pd.isnull(mouth[i,0]) == True :
        if i not in a:
            a.append(i)
    if pd.isnull(eye[i,0]) == True:
        if i not in a:
            a.append(i)
    if pd.isnull(eye[i,2]) == True:
        if i not in a:
            a.append(i)

final_labels = np.hstack((eye, mouth))

load = 0
for i in a:
    i = i-load
    final_labels = np.vstack((final_labels[:i, :], final_labels[i+1:, :]))
    data = np.vstack((data[:i, :], data[i+1:, :]))
    load = load+1

final_labels = final_labels*1.77

data_new = np.zeros((len(data[:,0]), 170, 170, 1))
for i in range(len(data[:])):
    data_new[i] = cv2.resize(data[i], (170, 170)).reshape(170, 170, 1)

x_train, x_test, y_train, y_test = train_test_split(data_new, final_labels, test_size=0.1, random_state=1)


rg = tf.keras.regularizers.l2(0.001)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(170, 170, 1), kernel_regularizer=rg),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='valid'),

    tf.keras.layers.Conv2D(64, (3,3), padding='valid', activation='relu', kernel_regularizer=rg),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3,3), padding='valid', activation='relu', kernel_regularizer=rg),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='valid'),

    tf.keras.layers.Conv2D(128, (3,3), padding='valid', activation='relu', kernel_regularizer=rg),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3,3), padding='valid', activation='relu', kernel_regularizer=rg),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='valid'),

    tf.keras.layers.Conv2D(256, (3,3), padding='valid', activation='relu', kernel_regularizer=rg),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3,3), padding='valid', activation='relu', kernel_regularizer=rg),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), strides=1, padding='valid'),

    tf.keras.layers.Conv2D(256, (3,3), padding='valid', activation='relu', kernel_regularizer=rg),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=rg),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(6, kernel_regularizer=rg)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    steps_per_epoch=80,
    epochs=16,
    validation_data=(x_test,y_test),
    validation_steps=25
)

tf.keras.layers.Conv