import numpy as np
import cv2
#import sys
#import glob
#import time
#import imutils
'''
with open('mtx.npy', 'rb') as f:
    mtx = np.load(f) #camera matrix

with open('dist.npy', 'rb') as f:
    dist = np.load(f) #distortion coefficients

dict = cv2.aruco.DICT_6X6_1000 # aruco marker dictionary
arucoDict = cv2.aruco.Dictionary_get(dict)
arucoParams = cv2.aruco.DetectorParameters_create()

markerLength = 1.0

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

        R, _ = cv2.Rodrigues(rvec)

        R = np.array(R)
        tvec = np.squeeze(np.array(tvec), axis=0).T

        T_CM = np.concatenate((R,tvec), axis=1)

        mat = np.zeros((4, 1))
        mat[3, 0] = 1
        T_CM = np.concatenate((T_CM,mat.T), axis=0)
        print(T_CM)
    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
def read_image():
    image = 'test_2.png'
    img = cv2.imread(image)
    mtx, dist, markerLength, arucoParams, arucoDict = aruco_params()
    aruco_transform(img, mtx, dist, markerLength, arucoParams, arucoDict)
    cv2.imshow('Image', img)
    #key = cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.waitKey(0)


def aruco_params():
    with open('mtx.npy', 'rb') as f:
        mtx = np.load(f) #camera matrix

    with open('dist.npy', 'rb') as f:
        dist = np.load(f) #distortion coefficients

    dict = cv2.aruco.DICT_6X6_1000 # aruco marker dictionary
    arucoDict = cv2.aruco.Dictionary_get(dict)
    arucoParams = cv2.aruco.DetectorParameters_create()

    markerLength = 1.0

    return mtx, dist, markerLength, arucoParams, arucoDict

def aruco_transform(img, mtx, dist, markerLength, arucoParams, arucoDict):
    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict,
	    parameters=arucoParams)

    if len(corners) > 0:
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)
        cv2.aruco.drawDetectedMarkers(img, corners)
        cv2.aruco.drawAxis(img, mtx, dist, rvec, tvec, 1.0)


        R, _ = cv2.Rodrigues(rvec)



        R = np.array(R)
        tvec = np.squeeze(np.array(tvec), axis=0).T

        T_CM = np.concatenate((R,tvec), axis=1)

        mat = np.zeros((4, 1))
        mat[3, 0] = 1
        T_CM = np.concatenate((T_CM,mat.T), axis=0)
        np.set_printoptions(suppress=True)
        print(T_CM)
        #print(np.rot2euler(R))

read_image()
