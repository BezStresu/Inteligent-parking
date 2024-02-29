# elegankco działa
#TODO zmierzyć czas działania, dodać równoległe działanie programu

import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import easyocr
import time

harcascade = 'ape_model/haarcascade_russian_plate_number.xml'

cap = cv2.VideoCapture(0)

cap.set(3, 640)     # width
cap.set(4, 480)     # height
min_area = 500      # minimum size of plate are that will be detected
count = 11

while True:
    success, img = cap.read()

    temp = img
    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)       # coordinates and size of rectangle that will be embracing the plate
            cv2.putText(img, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)     # text below rectangle

            img_roi = img[y: y+h, x:x+w]    # capture image of what is inside purple box
            cv2.imshow("roi", img_roi)      # side frame for image of plate

    cv2.imshow('Result', img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        start = time.time()
        gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
        reader = easyocr.Reader(['en'])
        result = reader.readtext(bfilter)
        stop = time.time()
        print(result)
        print(stop-start)
        # plt.figure(1)
        # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
        # plt.show()

    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

