from measurement_model import h
import numpy as np

def data_association(particles, W, z_t, Q):
    det_Q = np.linalg.det(Q)
    inv_Q = np.linalg.inv(Q)

    num_obs = z_t.shape[1]
    num_particles = particles.shape[1]
    num_landmarks = W.shape[1]

    psi = np.zeros((num_obs, num_landmarks, num_particles))
    Psi = np.zeros((num_obs, num_particles))
    #nu_vec = np.zeros((12,num_obs*num_particles*num_landmarks))
    #count = 0
    for i in range(num_obs):
        for m in range(num_particles):
            for k in range(num_landmarks):
                z_hat = h(particles[:, m], W, k) #Maybe optimize
                nu = z_t[:, i] - z_hat
                psi[i, k, m] =  1 / (2 * np.pi * np.sqrt(det_Q)) * np.exp(- 1 / 2 * np.dot(nu.T, inv_Q).dot(nu))
    Psi = np.max(psi, axis=1)
    #print(Psi)
    return Psi # num_obs x num_particles
