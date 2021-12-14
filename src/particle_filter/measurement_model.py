from math import sqrt, atan2
import numpy as np

# print("measurement model")

#x_t: [x,z,theta]
#W: [x,z,theta]


def euler2rot(angles):
    a, b, g = angles


    R_z = np.array([[np.cos(a), -np.sin(a), 0],[np.sin(a), np.cos(a), 0],[0, 0, 1]]) # Alpha rotation
    R_y = np.array([[np.cos(b), 0, np.sin(b)],[0, 1, 0],[-np.sin(b), 0, np.cos(b)]]) # Beta rotation
    R_x = np.array([[1, 0, 0],[0, np.cos(g), -np.sin(g)],[0, np.sin(g), np.cos(g)]]) # Gamma rotation

    return np.dot(R_z, R_y).dot(R_x)

def transform_mat(R, t):

    T = np.concatenate((R,t), axis=1)
    #print(T.shape)
    #print(np.array([[0, 0, 0, 1]]).shape)
    T = np.concatenate((T,np.array([[0, 0, 0, 1]])), axis=0)
    return T


def h(x_t, W, j):

    x, y, z, alpha, beta, gamma = x_t

    R_GC = euler2rot((alpha, beta, gamma))
    '''
    t = np.zeros((3, 1))
    t[0] = x
    t[1] = y
    t[2] = z
    '''
    T_GC = transform_mat(R_GC, np.array([[x, y, z]]).T) #NEGATE

    W_x, W_y, W_z, W_alpha, W_beta, W_gamma = W[j]

    R_GM = euler2rot((W_alpha, W_beta, W_gamma))
    T_GM = transform_mat(R_GM, np.array([[W_x, W_y, W_z]]).T)
    #print(T_GC)
    print(np.dot(np.linalg.inv(T_GC), np.array([[2, 0, 1, 1]]).T))
    #print(np.linalg.inv(T_GC))
    #print(T_GM)
    T_CM = np.dot(np.linalg.inv(T_GC), T_GM) 
    #T_CM = np.dot(T_GC, T_GM)
    return T_CM

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    #x_test = np.array([0, 0, 0.5, 0, np.pi, 20/360 * 2 * np.pi])
    #x_test = np.array([0, 0, 0.5, 0, , 20/360 * 2 * np.pi])
    #x_test = np.array([2, 0, 0.5, np.pi, 45/360 * 2 * np.pi, 0])
    x_test = np.array([2, 0, 0.5, np.pi, 45/360 * 2 * np.pi, -20/360 * 2 * np.pi])

    #x_test = np.array([0, 0, 0.5, np.pi, 0, -20/360 * 2 * np.pi])
    W = np.array([(0, 0, 4, 0, np.pi, 0)]) #MULTIPLY BY R
    print(h(x_test, W, 0))
