# działa jak należy
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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # have RGB form so that's why I needed to convert it

        # applying filtering and edged localization
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
        edged = cv2.Canny(bfilter, 30, 200)  # Edge detection with  Canny algorithm

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)  # find a rectangle shape
            if len(approx) == 4:  # need to test this with different plates pic
                location = approx
                break
        print(location)

        mask = np.zeros(gray.shape, np.uint8)  # creates a blank mask in grey image size
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)  # impose contours
        new_image = cv2.bitwise_and(img, img, mask=mask)  # impose mask on original pic

        (x, y) = np.where(mask == 255)  # finds place where pixels isn't zeros so places where pic isn't black
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]  # adding some buffer
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        stop = time.time()
        print('time:', stop - start)
        print(result)


        # text = result[0][-2]
        # print(text)

        #plt.show()

    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

