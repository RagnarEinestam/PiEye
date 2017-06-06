import cv2
from sklearn.externals import joblib
import numpy as np
import tools as tools
import time as time
import glob
from sklearn import metrics

clf = joblib.load('eyeContactCls.pkl')
pca = joblib.load('pca.pkl')
lda = joblib.load('lda.pkl')
scale = 1

def hold_out_wild(XWild, YWild, clf):  # Hold-out test
    y_pred = clf.predict(XWild)
    score_f1 = metrics.f1_score(YWild, y_pred, average='macro')
    mcc_score = metrics.matthews_corrcoef(YWild, y_pred)
    return score_f1, mcc_score

def startSpeedTest(img_scale = 1.0):

    ##TIME##
    extractFeatureTime = 0
    ldaTime = 0
    predictTime = 0
    ppTime = 0
    landmarksTime = 0
    cropTime = 0

    startfps = time.time()
    for file in glob.glob('.../*.jpg'):
        img = cv2.imread(file, -1)

        ppStart = time.time()
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        landmarksStart = time.time()
        landmarks = tools.get_landmarks_fast(img, img_scale)
        landmarksStop = time.time()
        landmarksTime = landmarksTime + (landmarksStop - landmarksStart)

        if (len(landmarks) > 0):
            cropStart = time.time()
            eyeImage_left, leftRect = tools.crop_eye_img(img, landmarks, 36, 39)
            eyeImage_right, rightRect = tools.crop_eye_img(img, landmarks, 42, 45)
            cropStop = time.time()
            cropTime = cropTime + (cropStop - cropStart)

            ppStop = time.time()
            ppTime = ppTime + (ppStop - ppStart)

            data = np.array([landmarks, [eyeImage_left, np.array(leftRect)], [eyeImage_right, np.array(rightRect)]])
            # Pupil_RightEye = tools.get_pupil_center(eyeImage_right, False)
            # Pupil_LeftEye = tools.get_pupil_center(eyeImage_left, False)

            efStart = time.time()
            featureVector, eyes = tools.extract_feature_vector_live(data)
            efStop = time.time()
            extractFeatureTime = extractFeatureTime + (efStop - efStart)

            # leftX, leftY = tools.get_pupil_ratio(data[1][0], Pupil_LeftEye[0], Pupil_LeftEye[1])
            # rightX, rightY = tools.get_pupil_ratio(data[2][0], Pupil_RightEye[0], Pupil_RightEye[1])
            # headPoseRatio = tools.get_head_pose_ratio(36, 45, 30, landmarks, True)

            featureVector = np.array(featureVector, dtype="float64")
            featureVector = featureVector.reshape(1, -1)

            # PUPIL
            # leftX = np.array(leftX)
            # leftY = np.array(leftY)
            # rightX = np.array(rightX)
            # rightY = np.array(rightY)

            # POSE
            # headPoseRatio = np.array(headPoseRatio)

            # PCA
            # featureVector = pca.transform(featureVector)

            # LDA
            ldaStart = time.time()
            featureVector = lda.transform(featureVector)
            ldaStop = time.time()
            ldaTime = ldaTime + (ldaStop - ldaStart)

            # PUPIL
            # featureVector = np.hstack((featureVector, np.atleast_2d(leftX).T))
            # featureVector = np.hstack((featureVector, np.atleast_2d(leftY).T))
            # featureVector = np.hstack((featureVector, np.atleast_2d(rightX).T))
            # featureVector = np.hstack((featureVector, np.atleast_2d(rightY).T))

            # POSE
            # featureVector = np.hstack((featureVector, np.atleast_2d(headPoseRatio).T))

            predictStart = time.time()
            eyeContact = clf.predict(featureVector)
            predictStop = time.time()
            predictTime = predictTime + (predictStop - predictStart)
            if eyeContact[0] == 1:
                print ("Person looked at the camera!")
            else:
                print '0'
                # cv2.imshow("mask_eye_img", eyes)

        ch = cv2.waitKey(1)
        if ch == 32:
            break
    cv2.destroyAllWindows()
    endfps = time.time()

    strFPS = (10 / (endfps - startfps))
    strLand = landmarksTime / 10
    strCrop = cropTime / 10
    strExtr =  extractFeatureTime / 10
    strLDA = ldaTime / 10
    strPred = predictTime / 10

    with open("log.txt", "a") as myfile:
        myfile.write("Scale: " + str(img_scale) + "\n")
        myfile.write('frames per second: ' + str(strFPS) + "\n")
        myfile.write("landmarks: " + str(strLand) + "\n")
        myfile.write("crop eyes: " + str(strCrop) + "\n")
        myfile.write("extract features: " + str(strExtr) + "\n")
        myfile.write("lda: " + str(strLDA) + "\n")
        myfile.write("predict: " + str(strPred) + "\n")
        myfile.write('\n')
        myfile.write('\n')
        myfile.write('\n')


def startSpeedAccuracyTest(img_scale = 1.0):

    Xlist = []
    Ylist = []
    missedCounter = 0

    for file in glob.glob('.../true/*.jpg'):
        print file
        img = cv2.imread(file, -1)
        img = cv2.resize(img, (640, 480))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        landmarks = tools.get_landmarks_fast(img, img_scale)

        if (len(landmarks) > 0):
            eyeImage_left, leftRect = tools.crop_eye_img(img, landmarks, 36, 39)
            eyeImage_right, rightRect = tools.crop_eye_img(img, landmarks, 42, 45)

            data = np.array([landmarks, [eyeImage_left, np.array(leftRect)], [eyeImage_right, np.array(rightRect)]])

            featureVector, eyes = tools.extract_feature_vector_live(data)

            cv2.imshow("img", eyes)

            featureVector = np.array(featureVector, dtype="float64")
            featureVector = featureVector.reshape(1, -1)

            # LDA
            featureVector = lda.transform(featureVector)

            Xlist.append(featureVector)
            Ylist.append(1)
        else:
            missedCounter = missedCounter +1

        ch = cv2.waitKey(1)
        if ch == 32:
            break

    for file in glob.glob('.../false/*.jpg'):
        print file
        img = cv2.imread(file, -1)
        img = cv2.resize(img, (640, 480))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        landmarks = tools.get_landmarks_fast(img, img_scale)

        if (len(landmarks) > 0):
            eyeImage_left, leftRect = tools.crop_eye_img(img, landmarks, 36, 39)
            eyeImage_right, rightRect = tools.crop_eye_img(img, landmarks, 42, 45)

            data = np.array([landmarks, [eyeImage_left, np.array(leftRect)], [eyeImage_right, np.array(rightRect)]])

            featureVector, eyes = tools.extract_feature_vector_live(data)

            cv2.imshow("img", eyes)

            featureVector = np.array(featureVector, dtype="float64")
            featureVector = featureVector.reshape(1, -1)

            # LDA
            featureVector = lda.transform(featureVector)

            Xlist.append(featureVector)
            Ylist.append(0)
        else:
            missedCounter = missedCounter +1

        ch = cv2.waitKey(1)
        if ch == 32:
            break

    print "Pos", Ylist.count(1)
    print "Neg", Ylist.count(0)
    print "Missed samples", missedCounter
    Xlist = np.reshape(Xlist, (1, len(Xlist))).T
    iw_f1, iw_mcc = hold_out_wild(Xlist, Ylist, clf)

    iw_info = '\nIn the wild: \nf1 ' + str(iw_f1) + '\nMCC ' + str(iw_mcc) + "\n"

    with open("log.txt", "a") as myfile:
        myfile.write("Scale: " + str(img_scale))
        myfile.write(iw_info)
        myfile.write('\n')

startSpeedAccuracyTest(img_scale=1)