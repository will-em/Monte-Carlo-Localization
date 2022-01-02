import numpy as np
from numpy.linalg.linalg import _pinv_dispatcher
from data_association import data_association
import matplotlib.pyplot as plt

class Particle_filter:

    def __init__(self, num_of_particles, landmarks, Q, boundaries, random_ratio):
        self.num_of_particles = num_of_particles
        self.M = num_of_particles#landmarks.shape[1]
        self.Q = Q
        self.map = landmarks               # x, y, z, yaw, pitch, roll
        self.weights = np.zeros((num_of_particles,1))
        self.boundaries = boundaries# [x_start, x_end, z_start, z_end]
        self.particles = np.zeros((6, self.num_of_particles))
        self.random_ratio = random_ratio

        self.particles[0, :] = np.random.randint(self.boundaries[0], high=self.boundaries[1], size= num_of_particles) # x-row
        self.particles[2, :] = np.random.randint(self.boundaries[2], high=self.boundaries[3], size= num_of_particles) # z-row

        self.particles[3, :] = np.pi * np.ones((1, num_of_particles)) # yaw
        self.particles[4, :] = 2 * np.pi * np.random.rand(num_of_particles) # pitch
        self.particles[5, :] = np.pi * np.random.rand(num_of_particles) - np.pi/2 # roll
        #fig = plt.figure()
        #self.ax = plt.axes(projection='3d')

        self.X_mean = np.mean(self.particles[0,:])
        self.Z_mean = np.mean(self.particles[2,:])

        self.pitch_mean = np.mean(self.particles[4,:])
        self.roll_mean = np.mean(self.particles[5,:])



    def step(self, obv):
        updated_particles = np.zeros((6, self.num_of_particles))

        Psi = data_association(self.particles, self.map, obv, self.Q) # num_obs x num_of_particles
        #print(np.max(Psi))
        weights = np.prod(Psi, axis=0)
        #print(weights)
        self.weights = weights
        #print(np.max(weights))

        CDF = np.cumsum(weights)
        CDF = CDF/CDF[-1]
        #plt.plot(CDF)
        #plt.show()
        #print(CDF)

        r_0 = 1 / self.M * np.random.rand()

        for m in range(self.M):
            #print(r_0 + (m-1)/self.M)
            i = np.where(CDF >= r_0 + (m-1)/self.M, CDF, np.inf).argmin()
            updated_particles[:, m] = self.particles[:, i]

        #print(np.mean(self.particles[2,:]))
        #print(np.mean(self.particles[4,:]))

        #self.particles[1,:]
        self.X_mean = np.mean(updated_particles[0,:])
        self.Z_mean = np.mean(updated_particles[2,:])

        self.pitch_mean = np.mean(updated_particles[4,:])
        self.roll_mean = np.mean(updated_particles[5,:])

        #RANDOMNESS
        num_of_random = int(self.random_ratio * self.num_of_particles)
        indices = np.random.randint(0, high=(self.num_of_particles - 1), size=num_of_random) #Unique??
        #print(np.mean(self.particles[0,:]))

        for i in indices:
            updated_particles[0, i] = np.random.randint(self.boundaries[0], high=self.boundaries[1]) # x-row
            updated_particles[2, i] = np.random.randint(self.boundaries[2], high=self.boundaries[3]) # z-row

            updated_particles[3, i] = np.pi  # yaw
            updated_particles[4, i] = 2 * np.pi * np.random.rand() # pitch
            updated_particles[5, i] = np.pi * np.random.rand() - np.pi/2 # roll

        self.particles = updated_particles

        self.particles[1,:] = 0.0
        self.particles[3,:] = np.pi


    def plot(self):
        X = self.particles[0, :]
        Z = self.particles[2, :]

        plt.xlim([self.boundaries[0], self.boundaries[1]])
        plt.ylim([self.boundaries[2], self.boundaries[3]])
        plt.scatter(X, Z, alpha = 0.03)
        plt.scatter(np.mean(self.particles[0,:]), np.mean(self.particles[2,:]))
        #plt.scatter(np.mean(self.particles[2,:]))


        #ax = plt.axes(projection='3d')
        #ax.scatter3D(X, Z, self.weights)
        #plt.scatter()
        #print(self.particles[1,:])
        #print(self.particles[3,:])
        plt.show()
        #print(self.particles)
