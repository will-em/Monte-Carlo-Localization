from math import sqrt, atan2
import numpy as np

# print("measurement model")

#x_t: [x,z,theta]
#W: [x,z,theta]

def predict_measurement(x_t, W, j):
    z_t = np.zeros(6,1)
    x, z, th = x_t

    t = np.array([x, 0, z]).T # [x 0 z]

    R = np.array([[np.cos(th), 0, np.sin(th)],[0, 1, 0],[-np.sin(th), 0, np.cos(th)]])

    T_g_to_cam = np.concatenate((R,t), axis=0)
    T_g_to_cam = np.concatenate((T_g_to_cam,np.array([0, 0, 0, 1])), axis=1)



    '''
    z_t = np.zeros(3,1)
    x_axisx = x_t[0]
    x_axis_y = x_t[1]
    z_t[0] = sqrt((W[j][x_axisx] - x_axisx)**2 + (W[j][x_axis_y] - x_axis_y)**2)
    z_t[1] = atan2(W[j][x_axis_y]- x_axis_y, W[j][x_axisx]- x_axisx)
    return z_t
    '''
