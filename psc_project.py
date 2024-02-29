import cv2
import matplotlib.pyplot as plt
img_name = 'base_image.jpg'
path = 'C:/Users/User/Desktop/AGH/5 sem/PSC/' + str(img_name)
new_name = 'IMAGE_TEST1'
img = cv2.imread(path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(1)
plt.imshow(cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB))
plt.show()
(thresh, im_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
plt.figure(2)
plt.imshow(cv2.cvtColor(im_bw, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite('C:/Users/User/Desktop/AGH/5 sem/PSC/' + str(new_name) + '.MIF', im_bw)
