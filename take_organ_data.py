import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

point = tf.keras.models.load_model('face_marks/')
face_cas = cv2.CascadeClassifier('D:\Devanshu\OpenCV\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

train = []
label = []
i = 0
while True:
    _, img = cap.read()

    #FACE DETECT
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    block = face_cas.detectMultiScale(img, 1.1, 4)
    if block == ():
        continue

    block = block[0]
    x,y,w,h = block[0], block[1], block[2]+block[0], block[3]+block[1]

    cv2.rectangle(img, (x, y), (w, h), (255,255,255), 2)
    #cv2.imshow('img', img)

    #LANDMARK DETECT
    mark = img[y:h, x:w]
    mark = cv2.resize(mark, (170, 170))
    den = mark.reshape(1, 170, 170, 1)
    pre = point.predict(den)[0].astype(int)

    cv2.circle(mark, (pre[0], pre[1]), 1, (255, 255, 255),2)
    cv2.circle(mark, (pre[2], pre[3]), 1, (255, 255, 255),2)
    cv2.circle(mark, (pre[4], pre[5]), 1, (255, 255, 255),2)
    #cv2.imshow('mark', mark)


    #ORGAN DETECT
    det_img = img[y:h, x:w]
    det_img = cv2.resize(det_img, (170, 170))

    L_eye = det_img[pre[1]-10:pre[1]+15, pre[0]-20:pre[0]+20]
    R_eye = det_img[pre[3]-14:pre[3]+15, pre[2]-20:pre[2]+20]
    Mouth = det_img[pre[5]-14:pre[5]+18, pre[4]-30:pre[4]+30]

    L_eye = cv2.resize(L_eye, (30, 30))
    R_eye = cv2.resize(R_eye, (30, 30))
    Mouth = cv2.resize(Mouth, (30, 30))

    _, L_eye = cv2.threshold(L_eye, 100, 255, cv2.THRESH_TRUNC)
    _, R_eye = cv2.threshold(R_eye, 100, 255, cv2.THRESH_TRUNC)
    _, Mouth = cv2.threshold(Mouth, 140, 255, cv2.THRESH_BINARY)

    #cv2.imshow('Left_eye', L_eye)
    #cv2.imshow('Right_eye', R_eye)
    cv2.imshow('Mouth', Mouth)

    i = i+1
    photo = Mouth
    train.append(photo)
    label.append(1)
    print(i)

    k = cv2.waitKey(1)
    if k == ord('s'):
        break

print('complete...............')
train = np.array(train)
label = np.array(label)
data = {'train':train, 'label':label}
sio.savemat('D:\Rocknessss\TRAIN_DATA\Train6.mat', data)