from Particle_filter import Particle_filter
from aruco_detect import read_image, test_image
import numpy as np
import cv2

if __name__=="__main__":
    W_1 = np.array([(0, 0, 4, 0, np.pi, 0)]) #right marker (double_marker.png)
    W_2 = np.array([(4, 0, 0, 0, -np.pi/2, 0)]) #left marker (double_marker.png)
    W = np.concatenate((W_1.T, W_2.T), axis=1)

    Q = 0.1 * np.eye(12)
    x_start = -5.0; x_end = 5.0; z_start = -5.0; z_end = 5.0
    boundaries = [x_start, x_end, z_start, z_end]
    #for i in range(10)
    PF = Particle_filter(500, W, Q, boundaries, 0.05)
    PF.plot()

    #video = cv2.VideoCapture('video_test.mov')

    while True:
        #ret, img = video.read()

        #if ret is False:
        #    break

        z_t = test_image()
        PF.step(z_t)
        PF.plot()
