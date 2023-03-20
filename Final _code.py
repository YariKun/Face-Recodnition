from keras.models import load_model
from keras_preprocessing.image import img_to_array
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import shutil

i = 0
face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('EmotionDetectionModel.h5')

class_labels=['Angry','Happy','Neutral','Sad','Surprise']

FILE_FOLDER = 'data' #в какую папку сложить скриншоты
if not os.path.exists(FILE_FOLDER): #если папки нет
    os.mkdir(FILE_FOLDER) #создать такую папку


vid_capture = cv2.VideoCapture(0)  #запись с камеры
if not vid_capture.isOpened(): #если камера не доступна
    print("Ошибка открытия видеофайла")
else:
    # здесь берем фреймы в секунду двумя вариантами из-за разных камер
    # всегда один из них вернет -1
    # Первый более новый
    fps = vid_capture.get(cv2.CAP_PROP_FPS) #берем фреймы в секунду первый вариант
    print('Фреймов в секунду: ', fps, 'FPS')

    frame_count = vid_capture.get(cv2.CAP_PROP_FRAME_COUNT) #берем фреймы в секунду второй вариант
    print('Частота кадров: ', frame_count)
    print('\n-----------------------------\nДля завершения нажмите "q" ...')
file_count = 0
while vid_capture.isOpened():
    # Метод vid_capture.read() возвращают кортеж, первым элементом является логическое значение
    # а вторым кадр
    ret, frame = vid_capture.read()
    if cv2.waitKey(25) & 0xFF == ord('q'):  # ждем q
        break
    if ret:
        if cv2.waitKey(25) & 0xFF == ord('q'): #ждем q
            break
        file_count += 1
        print('Кадр {0:01d}'.format(file_count)) #печатаем номер кадра в консоль
        writefile = FILE_FOLDER + '/frame_{0:01d}.jpeg'.format(file_count) #создаем путь до кадра
        cv2.imwrite(writefile, frame) #сохраняем кадр
        i += 1
        frame = cv2.imread('D:\Projects\Emotion_detection\without_flask\data\\' + "frame_" + str(i) + ".jpeg")

        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = classifier.predict(roi)[0]
                print(preds)
                label = class_labels[preds.argmax()]
                print(label)
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow('Look', frame)  # показываем кадр в окне

            else:
                cv2.putText(frame, 'No Face Found', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                plt.figure(figsize=(16, 16))
                plt.imshow(frame)
                cv2.imshow('Look', frame)  # показываем кадр в окне

    else:
        break

vid_capture.release() # Освободить объект захвата видео
shutil.rmtree('D:\Projects\Emotion_detection\without_flask\data\\')
sys.exit()