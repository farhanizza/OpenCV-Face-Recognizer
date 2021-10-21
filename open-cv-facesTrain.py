import os
import cv2 as cv
import numpy as np

people = ['Ben Affleck', 'Elon Musk',
          'Elton John', 'Mbappe', 'Ronaldo', 'Farhan']
DIR = r'C:\Farhan\BINUS\OpenCV\Photos\train-dataset'

face_default = cv.CascadeClassifier('face_default.xml')
features = []
labels = []


def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):

            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)

            if img_array is None:
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = face_default.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:

                faces_roi = gray[y:y+h, x:x+w]

                features.append(faces_roi)

                labels.append(label)


create_train()
print('Training Done ------------------------')

# print(f'Lenght of the features = {len(features)}')
# print(f'Lenght of the labels = {len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train the recognizer on the features list and the labels list
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('Labels.npy', labels)
