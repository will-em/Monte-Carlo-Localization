import numpy as np
import imutils
import cv2
import sys
import glob
import time

with open('mtx.npy', 'rb') as f:
    mtx = np.load(f) #camera matrix

with open('dist.npy', 'rb') as f:
    dist = np.load(f) #distortion coefficients

#image = 'aruco_test.png'
#img = cv2.imread(image)

dict = cv2.aruco.DICT_6X6_1000 # aruco marker dictionary
arucoDict = cv2.aruco.Dictionary_get(dict)
arucoParams = cv2.aruco.DetectorParameters_create()

markerLength = 1.0
#(corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict,
#	parameters=arucoParams)

#rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)

#cv2.aruco.drawDetectedMarkers(img, corners)
#cv2.aruco.drawAxis(img, mtx, dist, rvec, tvec, 1.0)


video = cv2.VideoCapture('leftright.mov')

while video.isOpened():
    ret, img = video.read()

    if ret is False:
        break

    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict,
	    parameters=arucoParams)

    if len(corners) > 0:
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)
        cv2.aruco.drawDetectedMarkers(img, corners)
        cv2.aruco.drawAxis(img, mtx, dist, rvec, tvec, 1.0)

        #print('rvec: ' + str(rvec))
        #print('tvec: ' + str(tvec))
        R, _ = cv2.Rodrigues(rvec)
#        print('Rodrigues: ' + str(R))

        T_CM = np.concatenate((R,tvec), axis=0)
        T_CM = np.concatenate((T_CM,np.array([0, 0, 0, 1])), axis=1)

        #print(np.array([np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)]))
        #print(np.rad2deg(y))

    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





#print('rvec: ' + str(rvec))
#print('tvec: ' + str(tvec))
#R, _ = cv2.Rodrigues(rvec)
#print('Rodrigues: ' + str(R))

#cv2.imshow('Estimated pose', img)
#key = cv2.waitKey(0)
