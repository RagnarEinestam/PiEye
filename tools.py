import cv2
import numpy as np
import os
import math
import dlib

predictor_path = "shape_predictor_68_face_landmarks.dat"
maskSize = (36, 96)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

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


BLOWUP_FACTOR = 1
RELEVANT_DIST_FOR_CORNER_GRADIENTS = 8*BLOWUP_FACTOR
dilationWidth = 1+2*BLOWUP_FACTOR #must be an odd number
dilationHeight = 1+2*BLOWUP_FACTOR #must be an odd number
dilationKernel = np.ones((dilationHeight,dilationWidth),'uint8')
writeEyeDebugImages=False
eyeCounter = 0

def get_pupil_ratio(img, pupilX, pupilY, quality):
    (h, w) = img.shape[:2]
    img = cv2.resize(img, (int(h * quality), int(w * quality)))

    height, width = img.shape
    pupilX = pupilX * quality
    pupilY = pupilY * quality

    if ((pupilX != 0) and (pupilY != 0)):
        ratioX = pupilX / float(width)
        ratioY = pupilY / float(height)
        return ratioX, ratioY
    else:
        return height / 2, width / 2


def get_head_pose_ratio(left, right, mid, landmarks, ret):
    ratio = 0
    if (ret):
        [R_x, R_y] = landmarks[right]
        [L_x, L_y] = landmarks[left]
        [N_x, N_y] = landmarks[mid]

        rightToNose = R_x - N_x
        noseToLeft = N_x - L_x

        if (rightToNose > noseToLeft):
            ratio = float(rightToNose) / noseToLeft
            ratio = ratio - 1
        else:
            ratio = (float(noseToLeft) / rightToNose)
            ratio = -ratio + 1
    return ratio


def get_files_wild(pathWild):
    ##Grab all files
    selectedFiles = []
    for root, dirs, files in os.walk(pathWild):
        for name in files:
            if name.endswith((".npy")):
                selectedFiles.append([root, name])
    selectedFiles = np.array(selectedFiles)
    ##Remove those with no face detected
    selectedFiles = selectedFiles[np.array([not '_FaceNotDetected' in x for x in selectedFiles[:, 1]])]

    print ('{} files are selected for the training'.format(len(selectedFiles)))

    return selectedFiles


def extract_feature_vector(data, quality=1.0):
    '''
    This function gets the data and builds the eye image and uses that as the main feature vector
    '''
    landmarks = data[0]
    Left_Eye_img = data[1][0]
    Right_Eye_img = data[2][0]
    Left_Eye_img = correct_individual_eye_image(Left_Eye_img, np.subtract(landmarks[36], data[1][1]),
                                                np.subtract(landmarks[39], data[1][1]), quality)
    Right_Eye_img = correct_individual_eye_image(Right_Eye_img, np.subtract(landmarks[42], data[2][1]),
                                                 np.subtract(landmarks[45], data[2][1]), quality)
    eyes = np.concatenate((Left_Eye_img, Right_Eye_img), axis=1)
    eyes = apply_mask(eyes)
    featurevector = eyes
    # We have already equalized the histogram of the eye image and I don't think we need to use scale anymore
    featurevector = featurevector.flatten()

    return featurevector

def extract_feature_vector_live(data, quality=1.0):
    '''
    This function gets the data and builds the eye image and uses that as the main feature vector
    '''
    landmarks = data[0]
    Left_Eye_img = data[1][0]
    Right_Eye_img = data[2][0]
    Left_Eye_img = correct_individual_eye_image(Left_Eye_img, np.subtract(landmarks[36], data[1][1]),
                                                np.subtract(landmarks[39], data[1][1]), quality)
    Right_Eye_img = correct_individual_eye_image(Right_Eye_img, np.subtract(landmarks[42], data[2][1]),
                                                 np.subtract(landmarks[45], data[2][1]), quality)
    eyes = np.concatenate((Left_Eye_img, Right_Eye_img), axis=1)
    eyes = apply_mask(eyes)
    featurevector = eyes
    # We have already equalized the histogram of the eye image and I don't think we need to use scale anymore
    featurevector = featurevector.flatten()

    return featurevector, eyes


def correct_individual_eye_image(image, L_corner, R_corner, scale):
    '''
    this function gets the eye region (cropped from the original image) and eye corners and rotates the eye and makes the eye image useful
    scale (or quality) is actually a scaling parameter. quality 1.0 processes the image as it is but lower quality (e.g. 0.5) reduce the resolution of the image before processing it.
    '''
    ##scale the image in case original image is too large
    (h, w) = image.shape[:2]

    image = cv2.resize(image, (int(h * scale), int(w * scale)))
    L_corner = L_corner * scale
    R_corner = R_corner * scale

    image = cv2.equalizeHist(image)

    # eyeline_size
    eyeline_size = math.hypot(R_corner[0] - L_corner[0], R_corner[1] - L_corner[1])

    # center of the eye in the image
    center = ((L_corner[0] + R_corner[0]) / 2, (L_corner[1] + R_corner[1]) / 2)

    # calculate slope of the eye line
    slope = math.degrees(np.arctan(float(L_corner[1] - R_corner[1]) / float(L_corner[0] - R_corner[0])))

    # calculate center of rotation (here is image center)
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    # rotate the image by slope degrees around the center
    M = cv2.getRotationMatrix2D(center, slope, 1.0)
    image = cv2.warpAffine(image, M, (w, h))

    # add ones
    points = np.array([R_corner, L_corner])
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    # transform points
    new_points = M.dot(points_ones.T).T
    #
    mar = int((new_points[0, 0] - new_points[1, 0]) / 2)
    image = image[int(new_points[0, 1]) - mar:int(new_points[0, 1]) + mar, int(new_points[1, 0]):int(new_points[0, 0])]

    (wi, hei) = image.shape[:2]

    if (hei <> 0) & (wi <> 0):
        image = cv2.resize(image, (maskSize[1] / 2, maskSize[0]))
    else:
        image = np.zeros((maskSize[0], maskSize[1] / 2), np.uint8)

    image = cv2.equalizeHist(image)

    return image


def apply_mask(img):
    img2 = cv2.imread('mask.png')  # logo

    rows, cols, channels = img2.shape
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # apply mask
    roi = img[0:rows, 0:cols]
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2gray, img2gray, mask=mask_inv)
    # Put logo in ROI and modify the main image

    dst = cv2.add(img1_bg, img2_fg)
    img[0:rows, 0:cols] = dst
    return img


def get_files_columbia(path):
    ##Grab all files
    selectedFiles = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith((".npy")):
                selectedFiles.append([root, name])
    selectedFiles = np.array(selectedFiles)

    ##Remove those with no face detected
    selectedFiles = selectedFiles[np.array([not '_FaceNotDetected' in x for x in selectedFiles[:, 1]])]

    print ('{} files are selected for the training'.format(len(selectedFiles)))

    return selectedFiles

def add_noise_img(im):
    (hh,ww)=np.shape(im)
    random_data = np.random.randn(hh,ww)
    noisyImg = im + 10.*random_data
    noisyImg[noisyImg<0]=0
    noisyImg[noisyImg>255]=255
    noisyImg=noisyImg.astype(np.uint8)
    return noisyImg


def process_file_name(name):
    print name
    underline_idx = [ind for ind, x in enumerate(name) if x == "_"]
    V_idx = [ind for ind, x in enumerate(name) if x == "V"]
    H_idx = [ind for ind, x in enumerate(name) if x == "H"]
    P_idx = [ind for ind, x in enumerate(name) if x == "P"]

    gazeAngle = np.array([int(name[underline_idx[2] + 1:V_idx[0]]), int(name[underline_idx[3] + 1:H_idx[0]])])
    headAngle = np.array([int(name[underline_idx[1] + 1:P_idx[0]])])
    return gazeAngle, headAngle

def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        #         raise TooManyFaces
        return []
    if len(rects) == 0:
        #         raise NoFaces
        return []
    return np.array([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def get_landmarks_fast(im, scale = 1):
    img = cv2.resize(im, (0, 0), fx=scale, fy=scale)
    rects = detector(img, 1)
    if len(rects) > 1:
        #         raise TooManyFaces
        return []
    if len(rects) == 0:
        #         raise NoFaces
        return []
    return np.array([[int(p.x / scale), int(p.y / scale)] for p in predictor(img, rects[0]).parts()])

def crop_eye_img(img,landmarks,LeftCornerIndex,RightCornerIndex):
    [R_x,R_y]=landmarks[RightCornerIndex]
    [L_x,L_y]=landmarks[LeftCornerIndex]
    #eyeline_size
    eyeline_size=math.hypot(R_x - L_x, R_y - L_y)
    #center of the eye in the image
    center=((L_x+R_x)/2,(L_y+R_y)/2)
    #cropping the image to a square around the eye (square center at the eye center and size is n% more than eyeline_size)
    n=0.15
    L=eyeline_size+n*eyeline_size
    return img[center[1]-int(L/2):center[1]+int(L/2),center[0]-int(L/2):center[0]+int(L/2)],[center[0]-int(L/2),center[1]-int(L/2)]


def get_pupil_center(gray, getRawProbabilityImage=False):
    gray = gray.astype('float32')
    if BLOWUP_FACTOR != 1:
        gray = cv2.resize(gray, (0,0), fx=BLOWUP_FACTOR, fy=BLOWUP_FACTOR, interpolation=cv2.INTER_LINEAR)

    IRIS_RADIUS = gray.shape[0]*.75/2 #conservative-large estimate of iris radius TODO: make this a tracked parameter--pass a prior-probability of radius based on last few iris detections. TUNABLE PARAMETER
    dxn = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3) #optimization opportunity: blur the image once, then just subtract 2 pixels in x and 2 in y. Should be equivalent.
    dyn = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    magnitudeSquared = np.square(dxn)+np.square(dyn)

    # ########### Pupil finding
    magThreshold = magnitudeSquared.mean()*.6 #only retain high-magnitude gradients. <-- VITAL TUNABLE PARAMETER
                    # The value of this threshold is critical for good performance.
                    # todo: adjust this threshold using more images. Maybe should train our tuned parameters.
    # form a bool array, unrolled columnwise, which can index into the image.
    # we will only use gradients whose magnitude is above the threshold, and
    # (optionally) where the gradient direction meets characteristics such as being more horizontal than vertical.
    gradsTouse = (magnitudeSquared>magThreshold) & (np.abs(4*dxn)>np.abs(dyn))
    lengths = np.sqrt(magnitudeSquared[gradsTouse]) #this converts us to double format
    gradDX = np.divide(dxn[gradsTouse],lengths) #unrolled columnwise
    gradDY = np.divide(dyn[gradsTouse],lengths)
#     debugImg(gradsTouse*255)
##    ksize = 7 #kernel size = x width and y height of the filter
##    sigma = 4
##    blurredGray = cv2.GaussianBlur(gray, (ksize,ksize), sigma, borderType=cv2.BORDER_REPLICATE)
##    debugImg(gray)
##    blurredGray = cv2.blur(gray, (ksize,ksize)) #x width and y height. TODO: try alternately growing and eroding black instead of blurring?
    #isDark = blurredGray < blurredGray.mean()
    isDark = gray< (gray.mean()*.8)  #<-- TUNABLE PARAMETER
    global dilationKernel
    isDark = cv2.dilate(isDark.astype('uint8'), dilationKernel) #dilate so reflection goes dark too
    isDark = cv2.erode(isDark.astype('uint8'), dilationKernel)
#     debugImg(isDark*255)
    gradXcoords =np.tile( np.arange(dxn.shape[1]), [dxn.shape[0], 1])[gradsTouse] # build arrays holding the original x,y position of each gradient in the list.
    gradYcoords =np.tile( np.arange(dxn.shape[0]), [dxn.shape[1], 1]).T[gradsTouse] # These lines are probably an optimization target for later.
    minXForPupil = 0 #int(dxn.shape[1]*.3)
##    #original method
#     centers = np.array([[phi(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords,IRIS_RADIUS) if isDark[cy][cx] else 0 for cx in range(dxn.shape[1])] for cy in range(dxn.shape[0])])
    #histogram method
    centers = np.array([[phi_with_hist(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords, IRIS_RADIUS) if isDark[cy][cx] else 0 for cx in xrange(minXForPupil,dxn.shape[1])] for cy in xrange(dxn.shape[0])]).astype('float32')
    # display outputs for debugging
#     centers = np.array([[phiTest(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords) for cx in range(dxn.shape[1])] for cy in range(dxn.shape[0])])
#     debugImg(centers)
    maxInd = centers.argmax()
    (pupilCy,pupilCx) = np.unravel_index(maxInd, centers.shape)
    pupilCx += minXForPupil
    pupilCy /= BLOWUP_FACTOR
    pupilCx /= BLOWUP_FACTOR
    if writeEyeDebugImages:
        global eyeCounter
        eyeCounter = (eyeCounter+1)%5 #write debug image every 5th frame
        if False:
            cv2.imwrite( "eyeGray.png", gray/gray.max()*255) #write probability images for our report
            cv2.imwrite( "eyeIsDark.png", isDark*255)
            cv2.imwrite( "eyeCenters.png", centers/centers.max()*255)
    if getRawProbabilityImage:
        return (pupilCx, pupilCy, centers)
    else:
        return (pupilCx, pupilCy)



class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def phi_with_hist(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords, IRIS_RADIUS):
    '''
    This function is taken from the source code of the Optimeyes project: https://github.com/LukeAllen/optimeyes
    '''
    vecx = gradXcoords-cx
    vecy = gradYcoords-cy
    lengthsSquared = np.square(vecx)+np.square(vecy)
    # bin the distances between 1 and IRIS_RADIUS. We'll discard all others.
    binWidth = 1 #TODO: account for webcam resolution. Also, maybe have it transform ellipses to circles when on the sides? (hard)
    numBins =  int(np.ceil((IRIS_RADIUS-1)/binWidth))
    bins = [(1+binWidth*index)**2 for index in xrange(numBins+1)] #express bin edges in terms of length squared
    hist = np.histogram(lengthsSquared, bins)[0]
    maxBin = hist.argmax()
    slop = binWidth
    valid = (lengthsSquared > max(1,bins[maxBin]-slop)) &  (lengthsSquared < bins[maxBin+1]+slop) #use only points near the histogram distance
    dotProd = np.multiply(vecx,gradDX)+np.multiply(vecy,gradDY)
    valid = valid & (dotProd > 0) # only use vectors in the same direction (i.e. the dark-to-light transition direction is away from us. The good gradients look like that.)
    dotProd = np.square(dotProd[valid]) # dot products squared
    dotProd = np.divide(dotProd,lengthsSquared[valid]) #make normalized squared dot products
##    dotProd = dotProd[dotProd > .9] #only count dot products that are really close
    dotProd = np.square(dotProd) # squaring puts an even higher weight on values close to 1
    return np.sum(dotProd) # this is equivalent to normalizing vecx and vecy, because it takes dotProduct^2 / length^2

