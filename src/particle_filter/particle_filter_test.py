from Particle_filter import Particle_filter
from aruco_detect import read_image, test_image
import numpy as np
import cv2

if __name__=="__main__":
    W_1 = np.array([(0, 0, 4, 0, np.pi, 0)]) #right marker (double_marker.png)
    W_2 = np.array([(4, 0, 0, 0, -np.pi/2, 0)]) #left marker (double_marker.png)
    W = np.concatenate((W_1.T, W_2.T), axis=1)

    Q = 0.001 * np.eye(12)
    x_start = -10.0; x_end = 10.0; z_start = -10.0; z_end = 10.0
    boundaries = [x_start, x_end, z_start, z_end]
    #for i in range(10)
    PF = Particle_filter(4000, W, Q, boundaries, 0.01)
    PF.plot()

    video = cv2.VideoCapture('video_test.mov')
    i = 0
    while True:
        i += 1
        ret, img = video.read()

        if ret is False:
            break

        if i > 55:
            z_t = read_image(img)

            #cv2.imshow('Image', img) #REMOVE COMMENT TO SHOW IMAGE
            #cv2.waitKey(0.5)
            #z_t = test_image()
            PF.step(z_t)
            PF.plot()
