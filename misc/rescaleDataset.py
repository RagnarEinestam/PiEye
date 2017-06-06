# This program takes face images in the "path" and converts them to the training data set
# by finding each eye, attaching them next to each other and creating a fixed size image (96x36)
# and masking the image of eyes. The program stores the negative and positive eye images in the
# "path+/eyepair/pos/" and "path+/eyepair/neg/" folders

import numpy as np
import cv2
import glob
import os
import dlib
import math

scale2m = .195
scale6m = .065
scale10m = .039
scale14m = .028
scale18m = .022

path = '/Users/Nyandu/Desktop/DataSet/'
pathFinished = '/Users/Nyandu/Desktop/DatasetGazeLock/'
predictor_path = "/Users/Nyandu/Desktop/EyePie/shape_predictor_68_face_landmarks.dat"
faces_folder_path = "faces"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


for m in range(46, 57):
    # Reads all jpg files in a directory
    if m <= 9:
        imagePath = path + '000' + str(m) + '/*.jpg'
        asd = '000' + str(m) + '/'
    else:
        imagePath = path + '00' + str(m) + '/*.jpg'
        asd = '00' + str(m) + '/'
    imgList = glob.glob(imagePath)

    if len(imgList) <= 0:
        print 'No Images Found'
    for imag in imgList:
        img = cv2.imread(imag, -1)

        img2m = cv2.resize(img, (0, 0), fx=scale2m, fy=scale2m)
        img6m = cv2.resize(img, (0, 0), fx=scale6m, fy=scale6m)
        img10m = cv2.resize(img, (0, 0), fx=scale10m, fy=scale10m)
        img14m = cv2.resize(img, (0, 0), fx=scale14m, fy=scale14m)
        img18m = cv2.resize(img, (0, 0), fx=scale18m, fy=scale18m)

        dname, fname = os.path.split(os.path.abspath(imag))

        cv2.imwrite(pathFinished + asd + "2m_" + fname, img2m)
        cv2.imwrite(pathFinished + asd + "6m_" + fname, img6m)
        cv2.imwrite(pathFinished + asd + "10m_" + fname, img10m)
        cv2.imwrite(pathFinished + asd + "14m_" + fname, img14m)
        cv2.imwrite(pathFinished + asd + "18m_" + fname, img18m)



