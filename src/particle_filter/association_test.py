import numpy as np

from data_association import data_association
from aruco_detect import read_image

z = read_image()
#print(z[:,0])
#print(z[:,1])


W_1 = np.array([(0, 0, 4, 0, np.pi, 0)]) #Right marker (double_marker.png)
W_2 = np.array([(4, 0, 0, 0, -np.pi/2, 0)]) #Left marker (double_marker.png)
W = np.concatenate((W_1.T, W_2.T), axis=1)
#print(W)

x_test = np.array([[-0.5, 0, -0.5, np.pi, -45/360 * 2 * np.pi, 0/360 * 2 * np.pi]]).T

#print(x_test.shape)
Q = np.eye(12)
Psi = data_association(x_test, W, z, Q)
print('Psi = ' + str(Psi))
