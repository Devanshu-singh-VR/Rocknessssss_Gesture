import cv2 as cv
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

left_eye_open = sio.loadmat('D:\Rocknessss\TRAIN_DATA\Train1.mat')['train']
left_eye_close = sio.loadmat('D:\Rocknessss\TRAIN_DATA\Train2.mat')['train']
mouth_steady = sio.loadmat('D:\Rocknessss\TRAIN_DATA\Train5.mat')['train']
mouth_smile = sio.loadmat('D:\Rocknessss\TRAIN_DATA\Train6.mat')['train']

img = mouth_smile[20,:]

_, img = cv.threshold(img, 0, 255, cv.THRESH_TRUNC) #90trunc 180binary
cv.imshow('img',img)
cv.waitKey(0)