import numpy as np
import imutils
import cv2
import sys

with open('mtx.npy', 'rb') as f:
    mtx = np.load(f) #camera matrix

with open('dist.npy', 'rb') as f:
    dist = np.load(f) #distortion coefficients

image = 'aruco_test.png'
img = cv2.imread(image)

dict = cv2.aruco.DICT_6X6_1000 # aruco marker dictionary
arucoDict = cv2.aruco.Dictionary_get(dict)
arucoParams = cv2.aruco.DetectorParameters_create()

(corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict,
	parameters=arucoParams)

markerLength = 1.0

rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)

cv2.aruco.drawDetectedMarkers(img, corners)
cv2.aruco.drawAxis(img, mtx, dist, rvec, tvec, 1.0)

print('rvec: ' + str(rvec))
print('tvec: ' + str(tvec))

cv2.imshow('Estimated pose', img)
key = cv2.waitKey(0)
