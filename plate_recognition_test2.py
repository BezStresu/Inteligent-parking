# comments will be marked by '#'
# notes will be marked by '# /text/ #'

# installation easyocr and imutils #

import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import easyocr

# adding pic to work on them

img = cv2.imread('captured_plates/scaned_img_10.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
plt.figure(1)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))  # function imshow() expect that image provided to it will
                                                    # have RGB form so that's why I needed to convert it

# applying filtering and edged localization
plt.figure(2)
edged = cv2.Canny(bfilter, 30, 200)  # Edge detection with  Canny algorithm
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))


keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None

for contour in contours:
    approx = cv2.approxPolyDP(contour, 8, True)    # find a rectangle shape
    if len(approx) == 4:                                         # need to test this with different plates pic
        location = approx
        break
print(location)

plt.figure(3)
mask = np.zeros(gray.shape, np.uint8)   # creates a blank mask in grey image size
new_image = cv2.drawContours(mask, [location], 0,255, -1)   # impose contours
new_image = cv2.bitwise_and(img, img, mask=mask)    # impose mask on original pic

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

plt.figure(4)
(x, y) = np.where(mask == 255)      # finds place where pixels isn't zeros so places where pic isn't black
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]  # adding some buffer
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

# the plate is now extracted

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
print(result)

text = result[0][-2]
print(text)

plt.show()
