import numpy as np
from numpy.linalg.linalg import _pinv_dispatcher
from data_association import data_association 
import matplotlib.pyplot as plt

class Particle_filter:

    def __init__(self, num_of_particles, landmarks, Q, boundaries):
        self.num_of_particles = num_of_particles
        self.M = landmarks.shape[1]
        self.Q = Q
        self.map = landmarks               # x, y, z, yaw, pitch, roll
        self.boundaries = boundaries# [x_start, x_end, z_start, z_end]
        self.particles = np.zeros((6, self.num_of_particles))

        self.particles[0, :] = np.random.randint(self.boundaries[0], high=self.boundaries[1], size= num_of_particles) # x-row
        self.particles[2, :] = np.random.randint(self.boundaries[2], high=self.boundaries[3], size= num_of_particles) # z-row

        self.particles[3, :] = np.pi * np.ones((1, num_of_particles)) # yaw
        self.particles[4, :] = 2 * np.pi * np.random.rand(num_of_particles) # pitch
        self.particles[5, :] = np.pi * np.random.rand(num_of_particles) - np.pi/2 # roll



    def step(self, obv):
        updated_particles = np.zeros((6, self.num_of_particles))

        Psi = data_association(self.particles, self.map, obv, self.Q) # num_obs x num_of_particles
        weights = np.prod(Psi, axis=0)

        CDF = np.cumsum(weights)

        r_0 = 1 / self.M * np.random.rand()

        for m in range(self.M):
            i = np.where(CDF >= r_0 + (m-1)/self.M, CDF, np.inf).argmin()
            updated_particles[:, m] = self.particles[:, i]

        #RANDOMNESS
        num_of_random = 0.1 * self.M

        indices = np.random.randint(0, high=self.M-1, size=num_of_random) #Unique??

        for i in indices:
            updated_particles[0, i] = np.random.randint(self.boundaries[0], high=self.boundaries[1]) # x-row
            updated_particles[2, i] = np.random.randint(self.boundaries[2], high=self.boundaries[3]) # z-row

            updated_particles[3, i] = np.pi  # yaw
            updated_particles[4, i] = 2 * np.pi * np.random.rand() # pitch
            updated_particles[5, i] = np.pi * np.random.rand() - np.pi/2 # roll

        self.particles = updated_particles 


    def plot(self):
        X = self.particles[0, :]
        Z = self.particles[2, :]
        plt.scatter(X, Z)
        plt.show()
        print(self.particles)



