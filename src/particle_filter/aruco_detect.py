import numpy as np
import cv2
#import sys
#import glob
#import time
#import imutils
'''
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
    image = 'double_marker.png'
    #image = 'left.png'
    img = cv2.imread(image)
    mtx, dist, markerLength, arucoParams, arucoDict = aruco_params()
    z = aruco_transform(img, mtx, dist, markerLength, arucoParams, arucoDict)
    #cv2.imshow('Image', img) #REMOVE COMMENT TO SHOW IMAGE
    #cv2.waitKey(0)
    return z


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
    np.set_printoptions(suppress=True)

    if len(corners) > 0:
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)
        cv2.aruco.drawDetectedMarkers(img, corners)

        num_obs = tvec.shape[0]

        z_t = np.zeros((12,num_obs))

        for i in range(0,num_obs):
            cv2.aruco.drawAxis(img, mtx, dist, rvec[i], tvec[i], 1.0)

            R, _ = cv2.Rodrigues(rvec[i])
            R = np.array(R)

            T_CM = np.concatenate((R,tvec[i].T), axis=1)
            T_CM = np.matrix.flatten(T_CM)

            z_t[:,i] = T_CM # Add T matrix to measurements


    else:
        z_t = np.zeros((12,0))

    return z_t

if __name__ == '__main__':
    z = read_image()
    print(z[:,0])
