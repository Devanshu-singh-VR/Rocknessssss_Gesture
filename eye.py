import cv2 as cv

dat = cv.VideoCapture(0)

while True:
    _, read = dat.read()
    read = cv.cvtColor(read,cv.COLOR_BGR2GRAY)
    _, R_eye = cv.threshold(read, 100, 255, cv.THRESH_BINARY)

    cv.imshow('img', read)
    cv.imshow('nat', R_eye)
    cv.waitKey(1)

