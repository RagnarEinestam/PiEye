import cv2
import dlib
from sklearn.externals import joblib
import numpy as np
import tools as tools

clf = joblib.load('eyeContactcls.pkl')
pca = joblib.load('pca.pkl')
lda = joblib.load('lda.pkl')
scale = 1

# Start camera image capturing
capture = cv2.VideoCapture(0)
ret, img = capture.read()
while capture.isOpened():
    ret, img = capture.read()
    if ret:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        landmarks = tools.get_landmarks_fast(img, .5) # Tweakable variable, affects speed and range.
        if (len(landmarks) > 0):
            eyeImage_left, leftRect = tools.crop_eye_img(img, landmarks, 36, 39)
            eyeImage_right, rightRect = tools.crop_eye_img(img, landmarks, 42, 45)

            data = np.array([landmarks, [eyeImage_left, np.array(leftRect)], [eyeImage_right, np.array(rightRect)]])

            # Pupil_RightEye = tools.get_pupil_center(eyeImage_right, False)
            # Pupil_LeftEye = tools.get_pupil_center(eyeImage_left, False)

            featureVector, eyes = tools.extract_feature_vector_live(data)


            # leftX, leftY = tools.get_pupil_ratio(data[1][0], Pupil_LeftEye[0], Pupil_LeftEye[1])
            # rightX, rightY = tools.get_pupil_ratio(data[2][0], Pupil_RightEye[0], Pupil_RightEye[1])
            # headPoseRatio = tools.get_head_pose_ratio(36, 45, 30, landmarks, True)

            featureVector = np.array(featureVector, dtype="float64")
            featureVector = featureVector.reshape(1, -1)

            #PUPIL
            # leftX = np.array(leftX)
            # leftY = np.array(leftY)
            # rightX = np.array(rightX)
            # rightY = np.array(rightY)

            #POSE
            # headPoseRatio = np.array(headPoseRatio)

            #PCA
            #featureVector = pca.transform(featureVector)

            #LDA
            featureVector = lda.transform(featureVector)

            #PUPIL
            # featureVector = np.hstack((featureVector, np.atleast_2d(leftX).T))
            # featureVector = np.hstack((featureVector, np.atleast_2d(leftY).T))
            # featureVector = np.hstack((featureVector, np.atleast_2d(rightX).T))
            # featureVector = np.hstack((featureVector, np.atleast_2d(rightY).T))

            #POSE
            # featureVector = np.hstack((featureVector, np.atleast_2d(headPoseRatio).T))

            eyeContact = clf.predict(featureVector)
            if eyeContact[0] == 1:
                print ("Person looked at the camera!")
            else:
                print '0'

            cv2.namedWindow('mask_eye_img', cv2.WINDOW_NORMAL)

            eyes = cv2.resize(eyes, (0, 0), fx=10, fy=10)

            w, h = eyes.shape
            cv2.resizeWindow('mask_eye_img', h, w+15)

            cv2.imshow("mask_eye_img", eyes)


    ch = cv2.waitKey(1)
    if ch == 32:
        break
capture.release()
cv2.destroyAllWindows()
