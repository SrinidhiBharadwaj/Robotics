'''
This file contains the class for update step of the Particle Filter
In the update step, only the particle weights get updated while the position of the particles
remain unchanged
'''
import numpy as np
from parameters import lidar_params
import pr2_utils as utils

class Update():
    def __init__(self, map):
        self.map_res = map.m_res
        self.xmin, self.xmax = map.m_xmin, map.m_xmax
        self.ymin, self.ymax = map.m_ymin, map.m_ymax
        self.map_ins = map #Map instance

    def update(self, mu, weights, l_ranges, l_angles, num_particles):
        """Update step of the Particle Filter
        mu or the particle states remain unchanges and the weights associated with the particles change as per the observation model
        Steps:
        1. Convert lidar to cartesian system (values are determined by the range and angles)
        2. Project lidar points onto the world frame (transformation determined by the particles' x, y and theta values)
        3. Obtain the correlation of the lidar points and the particles and choose the highest correlation values
        4. Use the correlation value as the observation model (probabilitstic and hence the softmax) and pick the weight with 
           the highest value
        """
        temp_weights = weights #To avoid reference updates (Thanks C for the "pointer")
        x, y = self.map_ins.convert_scan_to_cartesian(l_ranges, l_angles)
        cartesian_coordinates = np.array(list(zip(x, y))).reshape(2, -1)
        cartesian_coordinates = np.vstack((cartesian_coordinates, np.ones((2, cartesian_coordinates.shape[1]))))
        #Transform to robot body frame
        body_frame = self.map_ins.convert_to_body(cartesian_coordinates)
        particle_corr = np.zeros(num_particles)

        x_map = np.arange(self.xmin, self.xmax+self.map_res, self.map_res)
        y_map = np.arange(self.ymin, self.ymax+self.map_res, self.map_res)
        x_range, y_range = np.arange(-self.map_res * 2, self.map_res * 2 + self.map_res, self.map_res), \
                            np.arange(-self.map_res * 2, self.map_res * 2 + self.map_res, self.map_res)

        for particle in range(num_particles):
            mu_k = mu[:, particle]
            x_t_k, y_t_k, theta_t_k = mu_k[0], mu_k[1], mu_k[2]
            w_T_b = self.map_ins.get_body_to_world_transform(x_t_k, y_t_k, theta_t_k)
            #Convert body frame co ordinated to world frame co ordinates
            world_coordinates = w_T_b @ body_frame
            
            #Compute correlation
            x_world = world_coordinates[0, :]
            y_world = world_coordinates[1, :]
            occupied = np.vstack((x_world, y_world))
            
            corr = utils.mapCorrelation(self.map_ins.m_exp, x_map, y_map, occupied, x_range, y_range)
            particle_corr[particle] = np.max(corr)

       # print(particle_corr)
        #Observation model is considered as a softmax function
        #Weights are delta functions shifted
        #P_t+1 | t = (Ph * weights) / (sum(ph * weights))

        p_obs = self.softmax(particle_corr)
        numerator = p_obs * temp_weights
        denominator = np.sum(numerator)
        updated_weights = numerator/denominator

        return updated_weights

    def softmax(self, z):
        '''
        Helper function to calculate the softmax value
        '''
        return (np.exp(z)/np.sum(np.exp(z), axis=0))

    def resample(self, mu, weights, sys_flg=False):
        '''
        Stratified Resampling method
        To convert this into systematic resampling, set the sys_flag to True
        Implemented according to the algorithm in lecture slides
        '''
        num_particles = weights.shape[1]
        updated_mu = np.zeros((mu.shape[0], mu.shape[1]))
        updated_weights = np.array([1/num_particles ]* num_particles).reshape(1, num_particles)
        j, c = 0, weights[0, 0]

        if sys_flg:
            u = np.random.uniform(0, 1/num_particles)

        for k in range(num_particles):
            if not sys_flg:
                u = np.random.uniform(0, 1/num_particles)
            beta = u + k/num_particles
            while beta > c:
                j, c = j + 1, c + weights[0, j]
            
            updated_mu[:,k] = mu[:,j]
        return updated_mu, updated_weights