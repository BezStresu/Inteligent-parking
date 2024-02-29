import sys
import time
import cv2
import easyocr
from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi
import imutils
import numpy as np
import pandas as pd
import datetime
import re

# Harcascade
harcascade = 'ape_model/haarcascade_russian_plate_number.xml'
min_area = 500

#   Database
curr_path = 'current_list.xlsx'
reg_path = 'register.xlsx'
car_curr = None
car_register = None

#   Initialization of global variables and their default values
normal1 = cv2.imread('camera-icon.png')
bfilter1 = cv2.imread('camera-icon.png')
edged1 = cv2.imread('camera-icon.png')
isolated1 = cv2.imread('camera-icon.png')
cropped_image1 = cv2.imread('camera-icon.png')
plate_tab = 0
en_ex_flag = 0  # 0 is an entrance, 1 is Exit

#   OCR initialization
reader = easyocr.Reader(['en'])


#   FUNCTIONS

def extract_plate(input_string):
    processed_string = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    processed_string = processed_string.upper()
    if len(processed_string) >= 5:
        return processed_string
    else:
        return []


def load_car_data(name):
    match name:
        case 'cur':
            global car_curr
            try:
                car_curr = pd.read_excel(curr_path)
            except FileNotFoundError:
                car_curr = pd.DataFrame(columns=['Plate number', 'Arrival_time'])
            return len(car_curr)
        case 'reg':
            global car_register
            try:
                car_register = pd.read_excel(reg_path)
            except FileNotFoundError:
                car_register = pd.DataFrame(columns=['Plate number', 'Arrival time', 'Departure time'])
            return len(car_register)


def save_car_data(name):
    match name:
        case 'cur':
            global car_curr
            car_curr.to_excel(curr_path, index=False, engine='openpyxl')
        case 'reg':
            global car_register
            car_register.to_excel(reg_path, index=False, engine='openpyxl')


def add_car(registration_number, arrival_time, departure_time, name):
    match name:
        case 'cur':
            global car_curr
            if find_car(registration_number, name):
                pass
            else:
                new_car = pd.DataFrame({
                    'Plate number': [registration_number],
                    'Arrival time': [arrival_time]
                })
                car_curr = pd.concat([car_curr, new_car], ignore_index=True)
                save_car_data('cur')
        case 'reg':
            global car_register
            new_car = pd.DataFrame({
                'Plate number': [registration_number],
                'Arrival time': [arrival_time],
                'Departure time': [departure_time]
            })

            car_register = pd.concat([car_register, new_car], ignore_index=True)
            save_car_data('reg')


def find_car(registration_number, name):
    match name:
        case 'cur':
            global car_curr
            result = car_curr.loc[car_curr['Plate number'] == registration_number]
            if not result.empty:
                return result.to_dict(orient='records')[0]
            else:
                return None
        case 'reg':
            global car_register
            result = car_register.loc[car_register['Plate number'] == registration_number]
            if not result.empty:
                return result.to_dict(orient='records')[0]
            else:
                return None


def update_departure_time(registration_number, new_departure_time):
    global car_register
    new_departure_time = pd.to_datetime(new_departure_time)
    row_index = car_register.index[car_register['Plate number'] == registration_number].tolist()
    if row_index:
        new_departure_time = pd.to_datetime(new_departure_time)
        car_register.loc[row_index, 'Departure time'] = new_departure_time
        save_car_data('reg')


def remove_car(registration_number):
    global car_curr
    row_index = car_curr.index[car_curr['Plate number'] == registration_number].tolist()

    if row_index:
        car_curr = car_curr.drop(row_index)
        # save_car_data()


def close_app():
    save_car_data('cur')
    save_car_data('reg')
    time.sleep(1)
    exit()


def plate_reco(img):
    global normal1
    global bfilter1
    global edged1
    global isolated1
    global cropped_image1
    global reader
    global plate_tab
    global en_ex_flag
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction

    # applying filtering and edged localization
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection with  Canny algorithm

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    #
    location = None

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 8, True)  # find a rectangle shape
        if len(approx) == 4:  # need to test this with different plates pic
            location = [approx]
            break

    if not location:
        normal1 = img
        bfilter1 = cv2.cvtColor(bfilter, cv2.COLOR_BGR2RGB)
        edged1 = cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)
        isolated1 = cv2.imread('camera-icon.png')
        cropped_image1 = cv2.imread('camera-icon.png')

    else:
        mask = np.zeros(gray.shape, np.uint8)  # creates a blank mask in grey image size
        isolated = cv2.drawContours(mask, location, 0, 255, -1)  # impose contours
        isolated = cv2.bitwise_and(img, img, mask=mask)  # impose mask on original pic

        (x, y) = np.where(mask == 255)  # finds place where pixels isn't zeros so places where pic isn't black
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]  # adding some buffer

        # the plate is now extracted

        # reader = easyocr.Reader(['en'])
        result12 = reader.readtext(cropped_image)

        if result12:
            plate_tab = result12[0][1]
            plate_tab = extract_plate(plate_tab)
            if not en_ex_flag:
                add_car(plate_tab, datetime.datetime.now(), 0, 'reg')
            print(plate_tab)
        else:
            print(result12)

        # function imshow() expect that image provided to it will
        # have RGB form so that's why I needed to convert it
        normal1 = img
        bfilter1 = cv2.cvtColor(bfilter, cv2.COLOR_BGR2RGB)
        edged1 = cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)
        isolated1 = cv2.cvtColor(isolated, cv2.COLOR_BGR2RGB)
        cropped_image1 = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)


class GUI(QDialog):

    def __init__(self):
        super(GUI, self).__init__()

        loadUi('gui_v1.ui', self)

        global reader
        global normal1
        global bfilter1
        global edged1
        global isolated1
        global cropped_image1
        global en_ex_flag

        load_car_data('cur')
        load_car_data('reg')

        self.c_parking.setText('Number of cars on parking = 0')
        self.c_traf.setText('Number of cars in traffic')
        self.c_parked.setText('Number of parked cars')
        self.on_butt.clicked.connect(self.onClicked)
        self.capture_butt.clicked.connect(self.onCapture)
        self.close_butt.clicked.connect(close_app)
        self.sel_cam.currentTextChanged.connect(self.sel_cam_changed)

        self.icon_cam = cv2.imread('camera-icon.png')
        self.displayImage(self.icon_cam, 'cam_gate')
        self.displayImage(self.icon_cam, 'captured_photo')
        self.displayImage(self.icon_cam, 'plate_photo')
        self.displayImage(self.icon_cam, 'cam1')
        self.displayImage(self.icon_cam, 'cam2')
        self.displayImage(self.icon_cam, 'cam3')
        self.displayImage(self.icon_cam, 'cam4')

        self.take_photo_flag = 0

    def sel_cam_changed(self):
        print(self.sel_cam.currentText())

    def plate_detection(self, frame):
        plate_cascade = cv2.CascadeClassifier(harcascade)
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
        det_flag = 0
        for (x, y, w, h) in plates:
            area = w * h
            if area > min_area:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),
                              2)  # coordinates and size of rectangle that will be embracing the plate
                cv2.putText(frame, "Plate detected", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255),
                            2)  # text below rectangle
                det_flag = 1
        if det_flag:
            self.en_ex_text.setText('Plate detected')
        else:
            self.en_ex_text.setText('No plate detected')

        if self.take_photo_flag == 1:
            print('captured')
            normal = frame
            plate_reco(normal)
            if not plate_tab:
                self.plate_text.setText('Plate nr')
            else:
                temp = plate_tab
                self.plate_text.setText('Plate nr is: ' + str(temp))
                if not en_ex_flag:
                    add_car(plate_tab, datetime.datetime.now(), 0, 'cur')
                    cv2.imwrite("captured_plates/en_" + str(temp) + ".jpg", normal)
                else:
                    remove_car(plate_tab)
                    update_departure_time(plate_tab, datetime.datetime.now())
                    cv2.imwrite("captured_plates/ex_" + str(temp) + ".jpg", normal)

                save_car_data('cur')
                save_car_data('reg')

            self.take_photo_flag = 0
        return frame

    def cam_in_gate(self, frame0, frame1):
        global en_ex_flag
        if self.sel_cam.currentText() == 'Select camera':
            self.displayImage(self.icon_cam, 'cam_gate')
        elif self.sel_cam.currentText() == 'Entry':
            en_ex_flag = 0
            frame0 = self.plate_detection(frame0)
            self.photos()
            self.displayImage(frame0, 'cam_gate')
        elif self.sel_cam.currentText() == 'Exit':
            en_ex_flag = 1
            frame1 = self.plate_detection(frame1)
            self.photos()
            self.displayImage(frame1, 'cam_gate')

    def photos(self):
        self.displayImage(cropped_image1, 'plate_photo')

        if self.sel_view.currentText() == 'Normal':
            self.displayImage(normal1, 'captured_photo')
        elif self.sel_view.currentText() == 'Bifilter':
            self.displayImage(bfilter1, 'captured_photo')
        elif self.sel_view.currentText() == 'Edged':
            self.displayImage(edged1, 'captured_photo')
        elif self.sel_view.currentText() == 'Isolated':
            self.displayImage(isolated1, 'captured_photo')
        elif self.sel_view.currentText() == 'Cropped':
            self.displayImage(cropped_image1, 'captured_photo')
        else:
            self.displayImage(self.icon_cam, 'captured_photo')

    def lists(self):
        self.curr_list.setText(str(car_curr))
        self.register_list.setText(str(car_register))

    def cam_in_cams(self, frame, cam):
        match cam:
            case 'cam1':
                self.displayImage(frame, 'cam1')
            case 'cam2':
                self.displayImage(frame, 'cam2')
            case 'cam3':
                self.displayImage(frame, 'cam3')
            case 'cam4':
                self.displayImage(frame, 'cam4')

    def onClicked(self):
        cap0 = cv2.VideoCapture(1)
        cap1 = cv2.VideoCapture(0)
        # cap2 = cv2.VideoCapture(2)    # uncomment
        # cap3 = cv2.VideoCapture(3)    # uncomment

        self.on_butt.setEnabled(False)
        self.onDuty(cap0, cap1)

    def onCapture(self):
        self.capture_butt.setEnabled(True)
        self.take_photo_flag = 1

    def displayImage(self, img, cam_name):
        qformat = QImage.Format_RGB888
        match cam_name:
            case 'cam_gate':
                h = 300
                w = int(h * 1.78)  # 1.78 proporcja kamery testowej(wbudowana w laptop)
                img = QImage(img, img.shape[1], img.shape[0], qformat)
                img = img.scaled(w, h, aspectRatioMode=QtCore.Qt.KeepAspectRatio)
                img = img.rgbSwapped()
                self.cam_gate.setPixmap(QPixmap.fromImage(img))
                self.cam_gate.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            case 'captured_photo':
                w = 400
                h = int(w / 1.78)
                img = QImage(img, img.shape[1], img.shape[0], qformat)
                img = img.scaled(w, h, aspectRatioMode=QtCore.Qt.KeepAspectRatio)
                img = img.rgbSwapped()
                self.captured_photo.setPixmap(QPixmap.fromImage(img))
                self.captured_photo.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            case 'plate_photo':
                w = 351
                h = int(w / 1.78)
                img = QImage(img, img.shape[1], img.shape[0], qformat)
                img = img.scaled(w, h, aspectRatioMode=QtCore.Qt.KeepAspectRatio)
                img = img.rgbSwapped()
                self.plate_photo.setPixmap(QPixmap.fromImage(img))
                self.plate_photo.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            case 'cam1':
                w = 600
                h = int(w / 1.78)
                img = QImage(img, img.shape[1], img.shape[0], qformat)
                img = img.scaled(w, h, aspectRatioMode=QtCore.Qt.KeepAspectRatio)
                img = img.rgbSwapped()
                self.cam1.setPixmap(QPixmap.fromImage(img))
                self.cam1.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            case 'cam2':
                w = 600
                h = int(w / 1.78)
                img = QImage(img, img.shape[1], img.shape[0], qformat)
                img = img.scaled(w, h, aspectRatioMode=QtCore.Qt.KeepAspectRatio)
                img = img.rgbSwapped()
                self.cam2.setPixmap(QPixmap.fromImage(img))
                self.cam2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            case 'cam3':
                w = 600
                h = int(w / 1.78)
                img = QImage(img, img.shape[1], img.shape[0], qformat)
                img = img.scaled(w, h, aspectRatioMode=QtCore.Qt.KeepAspectRatio)
                img = img.rgbSwapped()
                self.cam3.setPixmap(QPixmap.fromImage(img))
                self.cam3.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            case 'cam4':
                w = 600
                h = int(w / 1.78)
                img = QImage(img, img.shape[1], img.shape[0], qformat)
                img = img.scaled(w, h, aspectRatioMode=QtCore.Qt.KeepAspectRatio)
                img = img.rgbSwapped()
                self.cam4.setPixmap(QPixmap.fromImage(img))
                self.cam4.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def onDuty(self, cap0, cap1, cap2=0, cap3=0):
        while cap0.isOpened() & cap1.isOpened():
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            # ret2, frame2 = cap2.read()
            # ret3, frame3 = cap3.read()
            self.lists()

            if ret0:
                self.cam_in_gate(frame0, frame1)
                self.cam_in_cams(frame0, 'cam1')
            else:
                print('cam 0 not working')

            if ret1:
                # print('cam 1 working')
                self.cam_in_cams(frame1, 'cam2')
            else:
                print('cam 1 not working')

            # if ret2:
            #     self.cam_in_gate(frame2, 'cam3')
            # else:
            #     print('cam 2 not working')
            #
            # if ret2:
            #     self.cam_in_gate(frame3, 'cam4')
            # else:
            #     print('cam 3 not working')

            cv2.waitKey()
        cap0.release()
        cap1.release()
        # cap2.release()
        # cap3.release()
        cv2.destroyAllWindows()


app = QApplication(sys.argv)
window = GUI()
window.show()
try:
    sys.exit(app.exec_())
finally:
    print('error')
