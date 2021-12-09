from math import sqrt, atan2
import numpy as np
# print("measurement model")

def predict_measurement(x_t, W, j):
    z_t = np.zeros(2,1)
    x_axisx = x_t[0]
    x_axis_y = x_t[1]
    z_t[0] = sqrt((W[j][x_axisx] - x_axisx)**2 + (W[j][x_axis_y] - x_axis_y)**2)
    z_t[1] = atan2(W[j][x_axis_y]- x_axis_y, W[j][x_axisx]- x_axisx)
    return z_t 