import cv2 as cv
import numpy
import os
import hashlib

camera = cv.VideoCapture(0)
# cek kamera
if not camera.isOpened():
    print("Camera is not open...Exiting")
    exit()

labels = ['Background']
for label in labels:
    if not os.path.exists(label):
        os.mkdir(label)

for folder in labels:
    count = 1
    print('Press S to start data collect ' + folder)
    userInput = input()

    if userInput != 's':
        print('Wrong ketword....')
        exit()

    while True:
        status, frame = camera.read()
        if not status:
            print('Frame is not open.....')
            break
        # convert to gray
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('Video', gray)
        gray = cv.resize(gray, (500, 500))
        # store
        if cv.waitKey(1) == ord('c'):
            cv.imwrite('C:/Farhan/BINUS/OpenCV/'+folder +
                       '/img '+str(count)+'.png', gray)
        count = count + 1
        # exit
        if cv.waitKey(1) == ord('q'):
            break
camera.release()
cv.destroyAllWindows()
