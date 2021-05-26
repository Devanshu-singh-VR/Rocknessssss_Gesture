import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

#left_eye_open = sio.loadmat('D:\Rocknessss\TRAIN_DATA\Train1.mat')
#left_eye_close = sio.loadmat('D:\Rocknessss\TRAIN_DATA\Train2.mat')
#right_eye_open = sio.loadmat('D:\Rocknessss\TRAIN_DATA\Train3.mat')
#right_eye_close = sio.loadmat('D:\Rocknessss\TRAIN_DATA\Train4.mat')
mouth_steady = sio.loadmat('D:\Rocknessss\TRAIN_DATA\Train5.mat')
mouth_smile = sio.loadmat('D:\Rocknessss\TRAIN_DATA\Train6.mat')

#train = np.vstack((left_eye_open['train'], left_eye_close['train'], right_eye_open['train'],
                   #right_eye_close['train'], mouth_steady['train'], mouth_smile['train']))
#label = np.hstack((left_eye_open['label'], left_eye_close['label'], right_eye_open['label'],
                   #right_eye_close['label'], mouth_steady['label'], mouth_smile['label']))

#train = np.vstack((left_eye_open['train'], left_eye_close['train'])).reshape(len(left_eye_open['label'][0])+len(left_eye_close['label'][0]), 30, 30, 1)
#label = np.hstack((left_eye_open['label']*0, left_eye_close['label']))[0]

train = np.vstack((mouth_steady['train'], mouth_smile['train'])).reshape(len(mouth_steady['label'][0])+len(mouth_smile['label'][0]), 30, 30, 1)
label = np.hstack((mouth_steady['label']*0, mouth_smile['label']))[0]


x_train, x_test, y_train, y_test = train_test_split(train, label, test_size=0.1, random_state=1)

rg = tf.keras.regularizers.l2(0.001)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='valid', input_shape=(30, 30, 1), kernel_regularizer=rg),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Conv2D(64, (3,3), padding='valid', activation='relu', kernel_regularizer=rg),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3,3), padding='valid', activation='relu', kernel_regularizer=rg),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Conv2D(256, (3,3), padding='valid', activation='relu', kernel_regularizer=rg),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=rg),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=rg)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    steps_per_epoch=int(np.sqrt(len(x_train[:,0]))),
    epochs=40,
    validation_data=(x_test, y_test),
    validation_steps=int(np.sqrt(len(x_test[:,0])))
)

print(model.predict(x_test))
print(y_test)
print(label)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()
model.save('Mouth/')