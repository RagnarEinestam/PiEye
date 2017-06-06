import numpy as np
import cv2
import glob
import dlib
import math

import tools as tools

scale = 1.0

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
NOSE_POINTS2=list(range(27, 31))
JAW_POINTS = list(range(0, 17))
JAW_POINTS_left=list(range(0, 4))
JAW_POINTS_right=list(range(13, 17))

path = '' # .jpg images used for classification
pathTo = '' # .jpg to .npy folder

predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
file_name = ""

def get_landmarks(im):
    '''
    Detect face landmarks and return the result as numpy array
    '''
    rects = detector(im, 1)
    if len(rects) > 1:
#         raise TooManyFaces
        return []
    if len(rects) == 0:
#         raise NoFaces
         return []
    return np.array([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def parseDB():
    for m in range(24, 57):
        # Reads all jpg files in a directory

        if m <= 9:
            imagePath = path + '000' + str(m) + '/*.jpg'
        else:
            imagePath = path + '00' + str(m) + '/*.jpg'

        imgList = glob.glob(imagePath)
        if len(imgList) <= 0:
            print 'No Images Found'
        for imag in imgList:
            file_name = pathTo+imag[35:len(imag)]
            print file_name
            img = cv2.imread(imag, -1)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            landmarks = get_landmarks(img)
            if len(landmarks) > 0:

                Left_Eye_img, leftRect = tools.crop_eye_img(img, landmarks, 36, 39)
                Right_Eye_img, rightRect = tools.crop_eye_img(img, landmarks, 42, 45)
                head_center = (np.mean(landmarks[NOSE_POINTS2][:, 0]), np.mean(landmarks[NOSE_POINTS2][:, 1]))
                head_left = (np.mean(landmarks[JAW_POINTS_left][:, 0]), np.mean(landmarks[JAW_POINTS_left][:, 1]))
                head_right = (np.mean(landmarks[JAW_POINTS_right][:, 0]), np.mean(landmarks[JAW_POINTS_right][:, 1]))

                Pupil_RightEye = tools.get_pupil_center(Right_Eye_img, False)
                Pupil_LeftEye = tools.get_pupil_center(Left_Eye_img, False)
                head_left = np.subtract(head_left, head_center)
                head_right = np.subtract(head_right, head_center)

                Pupil_RightEye_N1 = np.subtract(Pupil_RightEye, head_center)
                Pupil_LeftEye_N1 = np.subtract(Pupil_LeftEye, head_center)

                RightEyeSize = math.hypot(landmarks[[36]][0, 0] - landmarks[[39]][0, 0],
                                          landmarks[[36]][0, 1] - landmarks[[39]][0, 1])
                LeftEyeSize = math.hypot(landmarks[[42]][0, 0] - landmarks[[45]][0, 0],
                                         landmarks[[42]][0, 1] - landmarks[[45]][0, 1])

                AverageEyeSize = np.mean([RightEyeSize, LeftEyeSize])

                head_left_N = np.divide(head_left, AverageEyeSize)
                head_right_N = np.divide(head_right, AverageEyeSize)

                Pupil_RightEye_N2 = np.divide(Pupil_RightEye_N1, AverageEyeSize)
                Pupil_LeftEye_N2 = np.divide(Pupil_LeftEye_N1, AverageEyeSize)

                np.save(file_name, np.array(
                    [landmarks, [Left_Eye_img, np.array(leftRect)], [Right_Eye_img, np.array(rightRect)], Pupil_RightEye,
                     Pupil_LeftEye, head_center, head_left, head_right, Pupil_RightEye_N1, Pupil_LeftEye_N1,
                     AverageEyeSize, head_left_N, head_right_N, Pupil_RightEye_N2, Pupil_LeftEye_N2]))
                print 'Saved!'


def parseDB2():
    for img in glob.glob(path+'true/*.jpg'):

        file_name = pathTo + img[len(pathTo)+2:len(img) - 4]
        print file_name
        img = cv2.imread(img, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        landmarks = get_landmarks(img)
        if len(landmarks) > 0:
            Left_Eye_img, leftRect = tools.crop_eye_img(img, landmarks, 36, 39)
            Right_Eye_img, rightRect = tools.crop_eye_img(img, landmarks, 42, 45)
            head_center = (np.mean(landmarks[NOSE_POINTS2][:, 0]), np.mean(landmarks[NOSE_POINTS2][:, 1]))
            head_left = (np.mean(landmarks[JAW_POINTS_left][:, 0]), np.mean(landmarks[JAW_POINTS_left][:, 1]))
            head_right = (np.mean(landmarks[JAW_POINTS_right][:, 0]), np.mean(landmarks[JAW_POINTS_right][:, 1]))

            Pupil_RightEye = tools.get_pupil_center(Right_Eye_img, False)
            Pupil_LeftEye = tools.get_pupil_center(Left_Eye_img, False)
            head_left = np.subtract(head_left, head_center)
            head_right = np.subtract(head_right, head_center)

            Pupil_RightEye_N1 = np.subtract(Pupil_RightEye, head_center)
            Pupil_LeftEye_N1 = np.subtract(Pupil_LeftEye, head_center)

            right_eye_size = math.hypot(landmarks[[36]][0, 0] - landmarks[[39]][0, 0],
                                      landmarks[[36]][0, 1] - landmarks[[39]][0, 1])
            left_eye_size = math.hypot(landmarks[[42]][0, 0] - landmarks[[45]][0, 0],
                                     landmarks[[42]][0, 1] - landmarks[[45]][0, 1])

            mean_eye_size = np.mean([right_eye_size, left_eye_size])

            head_left_N = np.divide(head_left, mean_eye_size)
            head_right_N = np.divide(head_right, mean_eye_size)

            Pupil_RightEye_N2 = np.divide(Pupil_RightEye_N1, mean_eye_size)
            Pupil_LeftEye_N2 = np.divide(Pupil_LeftEye_N1, mean_eye_size)

            np.save(file_name, np.array(
                [landmarks, [Left_Eye_img, np.array(leftRect)], [Right_Eye_img, np.array(rightRect)],
                 Pupil_RightEye,
                 Pupil_LeftEye, head_center, head_left, head_right, Pupil_RightEye_N1, Pupil_LeftEye_N1,
                 mean_eye_size, head_left_N, head_right_N, Pupil_RightEye_N2, Pupil_LeftEye_N2]))
            print 'Saved!'

    for img in glob.glob(path+'false/*.jpg'):

        file_name = pathTo + img[len(pathTo)+2:len(img) - 4]
        print file_name
        img = cv2.imread(img, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        landmarks = get_landmarks(img)
        if len(landmarks) > 0:
            Left_Eye_img, leftRect = tools.crop_eye_img(img, landmarks, 36, 39)
            Right_Eye_img, rightRect = tools.crop_eye_img(img, landmarks, 42, 45)
            head_center = (np.mean(landmarks[NOSE_POINTS2][:, 0]), np.mean(landmarks[NOSE_POINTS2][:, 1]))
            head_left = (np.mean(landmarks[JAW_POINTS_left][:, 0]), np.mean(landmarks[JAW_POINTS_left][:, 1]))
            head_right = (np.mean(landmarks[JAW_POINTS_right][:, 0]), np.mean(landmarks[JAW_POINTS_right][:, 1]))

            Pupil_RightEye = tools.get_pupil_center(Right_Eye_img, False)
            Pupil_LeftEye = tools.get_pupil_center(Left_Eye_img, False)
            head_left = np.subtract(head_left, head_center)
            head_right = np.subtract(head_right, head_center)

            Pupil_RightEye_N1 = np.subtract(Pupil_RightEye, head_center)
            Pupil_LeftEye_N1 = np.subtract(Pupil_LeftEye, head_center)

            right_eye_size = math.hypot(landmarks[[36]][0, 0] - landmarks[[39]][0, 0],
                                      landmarks[[36]][0, 1] - landmarks[[39]][0, 1])
            left_eye_size = math.hypot(landmarks[[42]][0, 0] - landmarks[[45]][0, 0],
                                     landmarks[[42]][0, 1] - landmarks[[45]][0, 1])

            mean_eye_size = np.mean([right_eye_size, left_eye_size])

            head_left_N = np.divide(head_left, mean_eye_size)
            head_right_N = np.divide(head_right, mean_eye_size)

            Pupil_RightEye_N2 = np.divide(Pupil_RightEye_N1, mean_eye_size)
            Pupil_LeftEye_N2 = np.divide(Pupil_LeftEye_N1, mean_eye_size)

            np.save(file_name, np.array(
                [landmarks, [Left_Eye_img, np.array(leftRect)], [Right_Eye_img, np.array(rightRect)],
                 Pupil_RightEye,
                 Pupil_LeftEye, head_center, head_left, head_right, Pupil_RightEye_N1, Pupil_LeftEye_N1,
                 mean_eye_size, head_left_N, head_right_N, Pupil_RightEye_N2, Pupil_LeftEye_N2]))
            print 'Saved!'

parseDB()

