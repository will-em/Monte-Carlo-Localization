from Particle_filter import Particle_filter
from aruco_detect import read_image, test_image
import numpy as np
import cv2

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


if __name__=="__main__":
    W_1 = np.array([(0, 0, 4, 0, np.pi, 0)]) #right marker (double_marker.png)
    W_2 = np.array([(4, 0, 0, 0, -np.pi/2, 0)]) #left marker (double_marker.png)
    W = np.concatenate((W_1.T, W_2.T), axis=1)

    Q = 5.0 * np.eye(12)
    #Q[3,3] = 10.0
    #Q[7,7] = 1.0
    #Q[11,11] = 10.0
    #print(Q)
    x_start = -10.0; x_end = 10.0; z_start = -10.0; z_end = 10.0
    boundaries = [x_start, x_end, z_start, z_end]
    #for i in range(10)
    PF = Particle_filter(4000, W, Q, boundaries, 0.2)
    #PF.plot()

    video = cv2.VideoCapture('video_test.mov')

    #fig = plt.figure()
    #scatt = plt.scatter(PF.particles[0,:],PF.particles[2,:])

    i = 0
    while True:
        i += 1
        ret, img = video.read()

        if ret is False:
            break

        if i > 110:
            z_t = read_image(img)
            #cv2.imshow(img)
            fig = plt.figure()
            X = PF.particles[0,:]
            Z = PF.particles[2,:]
            plt.scatter(Z,X, alpha=0.03)

            X_mean = PF.X_mean
            Z_mean = PF.Z_mean
            plt.scatter(Z_mean,X_mean, label= 'X: ' + str(X_mean) + ', Z: ' + str(Z_mean))
            plt.legend(loc='upper right')
            plt.xlabel('z')
            plt.ylabel('x')

            fig.canvas.draw()

            minimap = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                    sep='')
            minimap  = minimap.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # img is rgb, convert to opencv's default bgr
            minimap = cv2.cvtColor(minimap,cv2.COLOR_RGB2BGR)

            cv2.imshow('Minimap', minimap)

            fig2 = plt.figure()
            pitch = np.rad2deg(PF.particles[4,:])
            roll = np.rad2deg(PF.particles[5,:])

            pitch_mean = np.rad2deg(PF.pitch_mean)
            roll_mean = np.rad2deg(PF.roll_mean)

            plt.scatter(pitch,roll, alpha = 0.3)
            plt.scatter(pitch_mean, roll_mean, label=('roll: ' + str(np.int(roll_mean)) + ', pitch: ' + str(np.int(pitch_mean))))
            plt.legend(loc='upper right')
            plt.xlabel('pitch')
            plt.ylabel('roll')

            fig2.canvas.draw()

            phasespace = np.fromstring(fig2.canvas.tostring_rgb(), dtype=np.uint8,
                    sep='')
            phasespace  = phasespace.reshape(fig2.canvas.get_width_height()[::-1] + (3,))

            # img is rgb, convert to opencv's default bgr
            phasespace = cv2.cvtColor(phasespace,cv2.COLOR_RGB2BGR)

            cv2.imshow('Phase', phasespace)

            cv2.imshow('Game', img) #shows game

            key = cv2.waitKey(33) & 0xFF
            if key == 27:
                break

            PF.step(z_t)
            #PF.plot()
