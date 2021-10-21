import cv2 as cv
import numpy as np
import os
from PIL import Image

face_cascade = cv.CascadeClassifier(
    'haarcascade/haarcascade_frontalface_alt2.xml')

# eye_cascade = cv.CascadeClassifier('haarcascade/haarcascade_eye.xml')

# smile_cascade = cv.CascadeClassifier('haarcascade/haarcascade_smile.xml')

recognizer = cv.face.LBPHFaceRecognizer_create()

recognizer.read('face_trained.yml')

people = ['Ben Affleck', 'Elon Musk',
          'Elton John', 'Mbappe', 'Ronaldo', 'Farhan']

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5)

    for (x, y, w, h) in faces:
        # print(x, y, w, h)

        roi_gray = gray[y:y+h, x:x+w]

        roi_color = frame[y:y+h, x:x+w]

        # recognizer
        id_, confidence = recognizer.predict(roi_gray)
        if confidence >= 45:
            print(id_)
            print(people[id_])
            font = cv.FONT_HERSHEY_SIMPLEX
            name = people[id_]
            color = (255, 255, 255)
            stroke = 2
            cv.putText(frame, name, (x+5, y-15), font,
                       1, color, stroke, cv.LINE_AA)

        img_item = 'my-image.png'
        cv.imwrite(img_item, roi_gray)

        color = (0, 255, 0)  # RGB
        stroke = 2
        width = x + w
        height = y + h

        cv.rectangle(frame, (x, y), (width, height), color, stroke)

        # eyes
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
        #     cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # smile
        # smile = smile_cascade.detectMultiScale(roi_gray)
        # for (xs, ys, ws, wh) in smile:
        #     cv.rectangle(roi_color, (xs, ys), (xs+ws, ys+wh), color, stroke)

    cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
