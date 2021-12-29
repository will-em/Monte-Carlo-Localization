import numpy as np
from numpy.linalg.linalg import _pinv_dispatcher
import data_association 

class particle_filter:

    def __init__(self, num_of_particles, landmarks, Q):
        self.num_of_particles = num_of_particles
        self.M = len(landmarks)
        self.Q = Q
        self.map = landmarks               # x, y, z, yaw, pitch, roll
        self.boundaries = [0, 100, 0, 100] # [x_start, x_end, z_start, z_end]
        self.particles = np.zeros((6, self.num_of_particles))

        self.particles[0, :] = np.random.randint(self.boundaries[0], high=self.boundaries[1], size= num_of_particles) # x-row
        self.particles[2, :] = np.random.randint(self.boundaries[2], high=self.boundaries[3], size= num_of_particles) # z-row

        self.particles[3, :] = np.pi * np.ones((1, num_of_particles)) # yaw
        self.particles[4, :] = 2 * np.pi * np.random.rand((1, num_of_particles)) # pitch
        self.particles[5, :] = np.pi * np.random.rand((1, num_of_particles)) - np.pi/2 # roll


    def step(self, obv):
        updated_particles = np.zeros((6, self.num_of_particles))

        Psi = data_association(self.particles, self.map, obv, self.Q) # num_obs x num_of_particles
        weights = np.prod(Psi, axis=0)
        pass


    def plot():
        pass



