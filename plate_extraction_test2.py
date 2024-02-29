import cv2

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
        cv2.imwrite("captured_plates/scaned_img_" + str(count) + ".jpg", img)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Captured", img)
        cv2.waitKey(500)
        count += 1
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

