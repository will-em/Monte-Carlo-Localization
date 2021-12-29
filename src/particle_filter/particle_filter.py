import numpy as np
import measurement_model as mm
import data_association as data

class particle_filter:

    def __init__(self, num_of_particles, landmarks):
        self.num_of_particles = num_of_particles
        self.M = len(landmarks)
        self.map = landmarks               # x y z yaw pitch roll
        self.boundaries = [0, 100, 0, 100] # [x_start, x_end, z_start, z_end]
        self.particles = np.zeros((6, num_of_particles))
    
        for i in range(num_of_particles):
            self.particles[:, i] = np.array()

    def step(self, obv):
        
        pass


    def plot():
        pass



