import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import tools as tools
import tests as tests

#Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB


path = '.../npy/' # .npy files for classifier training
pathWild = '.../npy2/' # .npy files for validation
maskSize = (36, 96)
n_components_lda = 1


def startTrain(trainFiles, trainFilesWild,image_quality, n_components, noise_neg, noise_pos,
               pca_state, lda_state, pupil_state, pose_state, wild_state):

    #CLASSIFIERS
    #clf = RandomForestClassifier(n_estimators=6, min_samples_split=6, min_samples_leaf=1, max_depth=1) #HYPERPARAMS
    clf = svm.SVC(gamma=.001, C=28, kernel='rbf', probability=True) #HYPERPARAMS
    #clf = LogisticRegression(C=1, fit_intercept=True, solver='sag') #HYPERPARAMS
    #clf = tree.DecisionTreeClassifier(min_samples_leaf=1, criterion='entropy', max_depth=1, presort=True) #HYPERPARAMS
    #clf = GradientBoostingClassifier(criterion='mae', n_estimators=1, loss='exponential', learning_rate=1, presort=True, max_depth=1) #HYPERPARAMS
    #clf = GaussianNB()

    Xlist = []
    Ylist = []
    pupilX = []

    pupilLeftX = []
    pupilLeftY = []
    pupilRightX = []
    pupilRightY = []
    headPoseRatio = []

    XlistWild = []
    YlistWild = []
    pupilXWild = []

    pupilLeftXWild = []
    pupilLeftYWild = []
    pupilRightXWild = []
    pupilRightYWild = []
    headPoseRatioWild = []

    if wild_state:

        if len(trainFilesWild) <= 0:
            print 'No .npy Found Wild'

        labelsWild = np.zeros((len(trainFilesWild),), dtype=np.int)

        for wild_i, wild in enumerate(trainFilesWild):

            name = str(wild)
            name = name[2:-2]
            name = name.replace("'", "").strip()
            name = name.replace(" ", "").strip()
            print name

            if 'true' in name:
                labelsWild[wild_i] = 1

            dat = np.load(name)

            if labelsWild[wild_i] == 1:

                XlistWild.append(tools.extract_feature_vector(dat, quality=1.0))
                YlistWild.append(1)

                if (pupil_state):
                    leftX, leftY = tools.get_pupil_ratio(dat[1][0], dat[4][0], dat[4][1], quality=1.0)
                    rightX, rightY = tools.get_pupil_ratio(dat[2][0], dat[3][0], dat[3][1], quality=1.0)

                    pupilLeftXWild.append(leftX)
                    pupilLeftYWild.append(leftY)
                    pupilRightXWild.append(rightX)
                    pupilRightYWild.append(rightY)

                    if (pose_state):
                        poseRatio = tools.get_head_pose_ratio(36, 45, 30, dat[0], True)
                        pupilXWild.append([leftX, leftY, rightX, rightY, poseRatio])
                    else:
                        pupilXWild.append([leftX, leftY, rightX, rightY])

                if (pose_state):
                    headPoseRatioWild.append(tools.get_head_pose_ratio(36, 45, 30, dat[0], True))

            else:

                XlistWild.append(tools.extract_feature_vector(dat, quality=1.0))
                YlistWild.append(0)

                if (pupil_state):
                    leftX, leftY = tools.get_pupil_ratio(dat[1][0], dat[4][0], dat[4][1], quality=1.0)
                    rightX, rightY = tools.get_pupil_ratio(dat[2][0], dat[3][0], dat[3][1], quality=1.0)

                    pupilLeftXWild.append(leftX)
                    pupilLeftYWild.append(leftY)
                    pupilRightXWild.append(rightX)
                    pupilRightYWild.append(rightY)

                    if (pose_state):
                        poseRatio = tools.get_head_pose_ratio(36, 45, 30, dat[0], True)
                        pupilXWild.append([leftX, leftY, rightX, rightY, poseRatio])
                    else:
                        pupilXWild.append([leftX, leftY, rightX, rightY])

                if (pose_state):
                    headPoseRatioWild.append(tools.get_head_pose_ratio(36, 45, 30, dat[0], True))

    if len(trainFiles) <= 0:
        print 'No .npy Found Wild'

    labels = np.zeros((len(trainFiles),), dtype=np.int)

    for file_i, file in enumerate(trainFiles):

        ##get label
        gaze_angle, head_angle = tools.process_file_name(file[1])
        if np.array_equal(gaze_angle, np.array([0, 0])):
            labels[file_i] = 1

        ##load file
        data = np.load(file[0] + file[1])

        ##we process this file for each quality given
        for q_i, q in enumerate(image_quality):

            if labels[file_i] == 1:
                ## keep a clean version of the image before adding noise to it
                eyeImage_left = data[1][0].astype(float)
                eyeImage_right = data[2][0].astype(float)

                for ccc in range(noise_pos):  # repeat the positive files k times and add noise each time
                    ##adding noise to the landmark coordinates
                    noise_mean = 0
                    noise_std = 2
                    noise = np.random.normal(noise_mean, noise_std, 2 * len(data[0])).reshape((len(data[0]), 2))

                    if ccc != 0:  # don't add noise first time
                        data[0] = np.add(data[0], noise).astype(int)
                        ##adding noise to the eye images
                        data[1][0] = tools.add_noise_img(eyeImage_left)
                        data[2][0] = tools.add_noise_img(eyeImage_right)

                    Xlist.append(tools.extract_feature_vector(data, quality=q))
                    Ylist.append(1)

                    if (pupil_state):
                        leftX, leftY = tools.get_pupil_ratio(data[1][0], data[4][0], data[4][1], q)
                        rightX, rightY = tools.get_pupil_ratio(data[2][0], data[3][0], data[3][1], q)

                        pupilLeftX.append(leftX)
                        pupilLeftY.append(leftY)
                        pupilRightX.append(rightX)
                        pupilRightY.append(rightY)

                        if (pose_state):
                            poseRatio = tools.get_head_pose_ratio(36, 45, 30, data[0], True)
                            pupilX.append([leftX, leftY, rightX, rightY, poseRatio])
                        else:
                            pupilX.append([leftX, leftY, rightX, rightY])

                    if (pose_state):
                        headPoseRatio.append(tools.get_head_pose_ratio(36, 45, 30, data[0], True))

            else:
                ## keep a clean version of the image before adding noise to it
                eyeImage_left = data[1][0].astype(float)
                eyeImage_right = data[2][0].astype(float)

                for ccc in range(noise_neg):  # repeat the positive files k times and add noise each time
                    ##adding noise to the landmark coordinates
                    noise_mean = 0
                    noise_std = 2
                    noise = np.random.normal(noise_mean, noise_std, 2 * len(data[0])).reshape((len(data[0]), 2))

                    if ccc != 0:  # don't add noise first time
                        data[0] = np.add(data[0], noise).astype(int)
                        ##adding noise to the eye images
                        data[1][0] = tools.add_noise_img(eyeImage_left)
                        data[2][0] = tools.add_noise_img(eyeImage_right)

                    Xlist.append(tools.extract_feature_vector(data, quality=q))
                    Ylist.append(0)

                    if (pupil_state):
                        leftX, leftY = tools.get_pupil_ratio(data[1][0], data[4][0], data[4][1], q)
                        rightX, rightY = tools.get_pupil_ratio(data[2][0], data[3][0], data[3][1], q)

                        pupilLeftX.append(leftX)
                        pupilLeftY.append(leftY)
                        pupilRightX.append(rightX)
                        pupilRightY.append(rightY)

                        if (pose_state):
                            poseRatio = tools.get_head_pose_ratio(36, 45, 30, data[0], True)
                            pupilX.append([leftX, leftY, rightX, rightY, poseRatio])
                        else:
                            pupilX.append([leftX, leftY, rightX, rightY])

                    if (pose_state):
                        headPoseRatio.append(tools.get_head_pose_ratio(36, 45, 30, data[0], True))

    print "Pos", Ylist.count(1)
    print "Neg", Ylist.count(0)

    if wild_state:
        print "pos wild", YlistWild.count(1)
        print "neg wild", YlistWild.count(0)

    if (pca_state == False) and (lda_state == False):
        if pupil_state:
            if (pose_state):
                print 'X is pupilX'
                X = np.array(pupilX)
                X = np.reshape(pupilX, (5, len(pupilX))).T
                if(wild_state):
                    XWild = np.array(pupilXWild)
                    XWild = np.reshape(pupilXWild, (5, len(pupilXWild))).T
            else:
                print 'X is pupilX'
                X = np.array(pupilX)
                X = np.reshape(pupilX, (4, len(pupilX))).T
                if(wild_state):
                    XWild = np.array(pupilXWild)
                    XWild = np.reshape(pupilXWild, (4, len(pupilXWild))).T
        else:
            print 'X is headPoseRatio'
            X = np.array(headPoseRatio)
            X = np.reshape(headPoseRatio, (1, len(headPoseRatio))).T
            if wild_state:
                XWild = np.array(headPoseRatioWild)
                XWild = np.reshape(headPoseRatioWild, (1, len(headPoseRatioWild))).T
    else:
        print 'X is Xlist'
        X = np.array(Xlist)
        if wild_state:
            XWild = np.array(XlistWild)

    Ylist = np.array(Ylist)
    if wild_state:
        YlistWild = np.array(YlistWild)

    if (pupil_state):
        if (pca_state or lda_state):
            pupilLeftX = np.array(pupilLeftX)
            pupilLeftY = np.array(pupilLeftY)
            pupilRightX = np.array(pupilRightX)
            pupilRightY = np.array(pupilRightY)
            if wild_state:
                pupilLeftXWild = np.array(pupilLeftXWild)
                pupilLeftYWild = np.array(pupilLeftYWild)
                pupilRightXWild = np.array(pupilRightXWild)
                pupilRightYWild = np.array(pupilRightYWild)
    if (pose_state):
        if (pca_state or lda_state):
            headPoseRatio = np.array(headPoseRatio)
            if wild_state:
                headPoseRatioWild = np.array(headPoseRatioWild)

    print "X", len(X)
    print "Y", len(Ylist)
    print "pupil", len(pupilX)
    print "Lx", len(pupilLeftX)
    print "Ly", len(pupilLeftY)
    print "Rx", len(pupilRightX)
    print "Ry", len(pupilRightY)
    print "Pose", len(headPoseRatio)

    if(wild_state):
        print "XWild", len(XWild)
        print "YWild", len(YlistWild)
        print "pupilWild", len(pupilXWild)
        print "LxWild", len(pupilLeftXWild)
        print "LyWild", len(pupilLeftYWild)
        print "RxWild", len(pupilRightXWild)
        print "RyWild", len(pupilRightYWild)
        print "PoseWild", len(headPoseRatioWild)

    if (pca_state):
        print 'pca fit ..'
        pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(X)
        X = pca.transform(X)
        joblib.dump(pca, 'pca.pkl')
        if wild_state:
            pcaWild = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(XWild)
            XWild = pcaWild.transform(XWild)

    if (lda_state):
        print 'lda fit ..'
        lda = LinearDiscriminantAnalysis(n_components=n_components_lda).fit(X, Ylist)
        X = lda.transform(X)
        joblib.dump(lda, 'lda.pkl')
        if wild_state:
            lda = LinearDiscriminantAnalysis(n_components=n_components_lda).fit(XWild, YlistWild)
            XWild = lda.transform(XWild)

    if (pupil_state):
        if (pca_state or lda_state):
            print 'added pupil to X'
            X = np.hstack((X, np.atleast_2d(pupilLeftX).T))
            X = np.hstack((X, np.atleast_2d(pupilLeftY).T))
            X = np.hstack((X, np.atleast_2d(pupilRightX).T))
            X = np.hstack((X, np.atleast_2d(pupilRightY).T))
            if wild_state:
                XWild = np.hstack((XWild, np.atleast_2d(pupilLeftXWild).T))
                XWild = np.hstack((XWild, np.atleast_2d(pupilLeftYWild).T))
                XWild = np.hstack((XWild, np.atleast_2d(pupilRightXWild).T))
                XWild = np.hstack((XWild, np.atleast_2d(pupilRightYWild).T))
    if (pose_state):
        if (pca_state or lda_state):
            print 'added pose to X'
            X = np.hstack((X, np.atleast_2d(headPoseRatio).T))
            if wild_state:
                XWild = np.hstack((XWild, np.atleast_2d(headPoseRatioWild).T))

    print 'clf fit ..'
    clf.fit(X, Ylist)
    joblib.dump(clf, 'eyeContactCls.pkl')

    if wild_state:
        return X, Ylist, XWild, YlistWild, clf, pca_state, lda_state, pupil_state, pose_state, wild_state
    else:
        return X, Ylist, 0, 0, clf, pca_state, lda_state, pupil_state, pose_state, wild_state

########################################################################################################################

trainFiles = tools.get_files_columbia(path)
trainFilesWild = tools.get_files_wild(pathWild)

X, Ylist, XWild, YlistWild, clf, pca_state, lda_state, pupil_state, pose_state, wild_state = startTrain(trainFiles, trainFilesWild, image_quality=[1],
                                                                              n_components=200, noise_neg=1, noise_pos=8,
                                                                          pca_state=False, lda_state=True,
                                                                          pupil_state=False, pose_state=False,
                                                                                            wild_state=False)

# tests.gridsearch_params(X, Ylist)

# print 'train/test split ..'
# tt_f1, tt_mcc = tests.hold_out_test(X, Ylist, clf)
# print 'cross validation ..'
# cv_f1, cv_mcc = tests.cross_fold_validation(X, Ylist, clf)
# print 'In the wild ..'
# iw_f1, iw_mcc = tests.hold_out_wild(X, Ylist, XWild, YlistWild, clf)
#
# run_info = 'PCA = ' + str(pca_state) + '\nLDA = ' + str(lda_state) + '\nPUPIL = ' + str(pupil_state) + '\nPOSE = ' + str(pose_state) + '\n'
# tt_info = '\nTrain/Test split: \nf1 ' + str(tt_f1) + '\nMCC ' + str(tt_mcc) + '\n'
# cv_info = '\ncross fold validation: \nf1 ' + str(cv_f1) + '\nMCC ' + str(cv_mcc) +'\n'
# iw_info = '\nIn the wild: \nf1 ' + str(iw_f1) + '\nMCC ' + str(iw_mcc)
#
# with open("log.txt", "a") as myfile:
#     myfile.write(run_info)
#     myfile.write(tt_info)
#     myfile.write(cv_info)
#     myfile.write(iw_info)
#     myfile.write('\n')
#     myfile.write('\n')
#     myfile.write('\n')