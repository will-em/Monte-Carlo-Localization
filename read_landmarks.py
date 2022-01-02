#from particle_filter.particle_filter_test import W
import re
import numpy as np

'''
1 = NORTH, YAW = np.pi, x+=0.5 z+=0.0
2 = EAST, YAW = np.pi / 2, x-=1.0, z+=0.5
3 = SOUTH, YAW = 0, x+=0.5, z-=1.0
4 = WEST, YAW = -np.pi / 2, x+=0.0, z+=0.5
'''

def read_landmarks():
    
    W_list = []

    with open('labyrinth_markers.txt') as f:
        
        lines = f.readlines()

        for line in lines:
            if line[0]=="(":
                test = re.findall(r'\d+', line)
                numbers = [int(s) for s in test]
                W = np.zeros(6)
                W[0] = numbers[0]
                W[2] = numbers[1]
                if numbers[2] == 1: # North
                    W[0] += 0.5
                    W[4] = np.pi

                elif numbers[2] == 2: # East
                    W[0] -= 1.0
                    W[2] += 0.5
                    W[4] = np.pi/2

                elif numbers[2] == 3: # South
                    W[0] += 0.5
                    W[2] -= 1.0
                    W[4] = 0.0
                    
                elif numbers[2] == 4: # West
                    W[2] += 0.5
                    W[4] = -np.pi/2

                W_list.append(W)

    # Convert to numpy array
    W = np.zeros((6, len(W_list)))
    for i, landmark in enumerate(W_list):
        #print(W[:, i].shape)
        W[:, i] = landmark 
        print(landmark)

    return W 