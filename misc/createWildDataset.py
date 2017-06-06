import cv2
import time

# pathTrue = '/Users/Nyandu/Desktop/InTheWildDataset/true/'
# pathFalse = '/Users/Nyandu/Desktop/InTheWildDataset/false/'
#
# capture = cv2.VideoCapture(0)
# ret, img = capture.read()
#
# imgPos = True
# counterTrue = 0
# counterFalse = 0
#
# while capture.isOpened():
#     ret, img = capture.read()
#     cv2.imshow("hej", img)
#
#     if ret:
#         if (imgPos):
#             file_name = pathTrue + "truePic"+str(counterTrue)+'.jpg'
#             print file_name
#             cv2.imwrite(file_name, img)
#             counterTrue = counterTrue +1
#         else:
#             file_name = pathFalse + "falsePic"+str(counterFalse)+'.jpg'
#             print file_name
#             counterFalse = counterFalse +1
#             cv2.imwrite(file_name, img)
#
#     time.sleep(.5)
#     ch = cv2.waitKey(1)
#     if ch == 32:
#         break
# capture.release()
# cv2.destroyAllWindows()

path = '/Users/Nyandu/Downloads/eyscrop.png'
img = cv2.imread(path, -1)
img = cv2.resize(img, (0, 0), fx=1, fy=1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.equalizeHist(img)
cv2.imwrite(path, img)