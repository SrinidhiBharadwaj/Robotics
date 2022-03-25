"""
Main file that contains SLAM operation
Utilizes Predict, Update and Map classes along with utilities functions to localize and map the surroundings of 
an autonomous car
Sensors: Encoder, Lidar, FOG
Motion model: Differential drive
Author: Srinidhi Kalgundi Srinivas
"""

import pickle
import numpy as np
from matplotlib import pyplot as plt
from parameters import lidar_params, fog_params, enc_params, stereo_cam_params
import pr2_utils as utils
import pandas as pd
import mapping as map
import prediction as pred
import update as update
import texture as texture
import cv2
from tqdm import tqdm

class Particle_Filter_SLAM():

    def __init__(self, lidar_data, num_particles=10):
        self.num_particles = num_particles
        self.weights = np.array([1/self.num_particles]*self.num_particles).reshape(1, self.num_particles)
        self.mu = np.zeros((3, self.num_particles)) # 3 x num_particles
        self.idx_max = np.argmax(self.weights)
        self.x_t = self.mu[:, self.idx_max] #Take the column with max index of mu, not required in init but adding for completeness
        self.n_threshold = 5
        self.dead_reckon = False

        self.map_env = map.Mapping(lidar_data)
        self.motion = pred.Prediction()
        #Passing map to update and texture class are essential as they use parameters of the map
        self.observe = update.Update(self.map_env)
        self.texture = texture.Texture(self.map_env)

    def SLAM(self, enc_data, lid_data, fog_data, cam_time):
        '''
        Similar to the main function of the project
        Loops through the encoder time and lidar time to perfect prediction and update step of the Particle Filter
        Provides calls to relevant functions and contains the SLAM pipeline
        Logic to decide whether to do the prediction or update is based on the timestamp value
        If encoder timestamp is lesser than lidar timestamp - Predict
        Else Update
        '''
        enc_time = enc_data[0]
        enc_data = enc_data[1]
        lidar_time = lid_data[0]
        lidar_data = lid_data[1]
        fog_time = fog_data[0]
        fog_data = fog_data[1]
        
        #particle_trajectory is a 2xN matrix containing x and y coordinates of the particles with highest -
        #-weights after every update step
        particle_trajectory = np.array([[0],[0]])

        #Number of iterations for particle filter (as per the algorithm: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa13/slides/bayes-filters.pdf
        # Slide 13 - observation time + update time
        num_iterations = np.size(enc_time) + np.size(lidar_time)
        l_idx, e_idx = 0, 1
       
        #Boolean flag to help with testing (not used for the final code)
        texture_test = False
        #Flag to overcome the previous flag (helped with debugging)
        plot_texture = False

        if self.dead_reckon:
            num_iterations = np.size(enc_time)
            self.num_particles = 1
            self.weights = np.array([1/self.num_particles]*self.num_particles).reshape(1, self.num_particles)
            self.mu = np.zeros((3, self.num_particles)) # 3 x num_particles

        for step in tqdm(range(num_iterations)): 
            if self.dead_reckon and e_idx < np.size(enc_time):
                mu_updated = self.predict(fog_time,fog_data, enc_time, enc_data, e_idx)
                particle_trajectory = np.hstack((particle_trajectory, mu_updated[0:2].reshape(2, 1)))
                #Update encoder and lidar indices
                if (e_idx < (np.size(enc_data)- 1)):
                    e_idx = e_idx+1
            else:
                #Essentail check to ensure no out of bounds cases
                if(l_idx < np.size(lidar_time) and e_idx < np.size(enc_time)):
                    #Condition to check whether predict or update needs to be performed
                    #If encoder timestamp is lesser than lidar time stamp, perform prediction
                    if (l_idx == np.size(lidar_time) or enc_time[e_idx] < lidar_time[l_idx]):
                        #Return value is only for debugging
                        mu_updated = self.predict(fog_time,fog_data, enc_time, enc_data, e_idx)
                        #Update encoder and lidar indices
                        if (e_idx < (np.size(enc_data)- 1)):
                            e_idx = e_idx+1
                        else:
                            l_idx = l_idx+1
                    
                    #Update step
                    else:
                        #Update weights of the particles based on the lidar scan at time t
                        l_ranges, l_angles = self.map_env.get_valid_lidar_ranges(l_idx)
                        self.update(l_ranges, l_angles)

                        #Pick the particle with the highest weight and follow its trajectory
                        best_particle = self.mu[:, np.argmax(self.weights)]
                        particle_trajectory = np.hstack((particle_trajectory, best_particle[0:2].reshape(2, 1)))
                        #print(best_particle)
                        self.update_map(best_particle, l_ranges, l_angles)

                        #Update texture later
                        #Disparity images is calculated from stereo_left and stereo_right images 
                        #function "create_disparity_map()" in pr2_utils.py"
                        #print(cam_time.shape)
                        if texture_test or plot_texture:
                            t_map = self.add_texture(best_particle, cam_time, lidar_time[l_idx])
                        
                        #Resample particles
                        self.resample_particles()

                        if (l_idx < np.size(lidar_time)):
                            l_idx += 1
                        else:
                            e_idx += 1
            
            #Save the map after 1000 iteration
            if step % 10000 == 0:
                print("Saving the map after {0} steps".format(step))
                if not texture_test:
                    map_result = ((1-1/(1+np.exp(self.map_env.m))) < 0.1).astype(np.int)
                    self.save_map(map_result, particle_trajectory, str(step), texture_map=self.map_env.t_map)

        #Save the final map
        map_result = ((1-1/(1+np.exp(self.map_env.m))) < 0.1).astype(np.int)
        self.save_map(map_result, particle_trajectory, texture_map=self.map_env.t_map, bin_result=None)
        plt.scatter(particle_trajectory[0,:], particle_trajectory[1,:], c='b')
        plt.show()

    def save_map(self, map_result, particle_trajectory, step="final", texture_map=None, bin_result=None):
        '''
        Helper function to plot and save the maps and particle trajectory
        '''
        xis = particle_trajectory[0,:]
        yis = particle_trajectory[1,:]
        xis = np.ceil((xis - self.map_env.m_xmin) / self.map_env.m_res ).astype(np.int16) - 1 
        yis = np.ceil((yis - self.map_env.m_ymin) / self.map_env.m_res ).astype(np.int16) - 1
        indGood = np.logical_and(np.logical_and(np.logical_and((xis >=0), (yis>=0)), (xis < self.map_env.num_cells_x)), 
                                            (yis < self.map_env.num_cells_y))


        map_result[xis[indGood],yis[indGood]] = 2
        if not self.dead_reckon:
            plt.scatter(yis,xis,marker='.', c = 'g',s = 0.1)
            plt.imshow(map_result, cmap = "hot")
            plt.savefig('map/'+(step)+'.png', format = 'png')
            
            plt.imsave('map_gray/'+(step)+'.png',map_result,cmap='gray')

            if texture_map is not None:
                #texture_map[xis[indGood],yis[indGood],:] = np.array([0,0,0])
                plt.imsave('texturemap/'+str(step)+'.png',texture_map)

            with open("map_parameters.pkl", 'wb') as f:
                pickle.dump([map_result, texture_map, xis, yis, self.mu, self.weights] , f)
                
            plt.close()
        else:
            plt.scatter(particle_trajectory[0,:],particle_trajectory[1,:],marker='.', c = 'g',s = 2)
            plt.savefig('map/'+(step)+'.png', format = 'png')
            plt.close()

    def resample_particles(self):
        '''
        Hook function to call the stratified resampling function in 
        the update class
        '''
        n_eff = 1 / np.sum(self.weights)
        if n_eff < self.n_threshold:
            self.observe.resample(self.mu, self.weights)


    def update_map(self, particles, l_ranges, l_angles):
        '''
        Hook function to calculate and update the log odds map
        Comments are only for my future reference
        '''
        updated_map = self.map_env.update_map(particles, l_ranges, l_angles)
       

    def predict(self, fog_time,fog_data, enc_time, enc_data, e_idx):
        '''
        Hook function to predict the state of particles at time t+1 using the motion model
        described in the prediction class
        '''
        mu_updated = self.motion.motion_model(self.mu, fog_time,fog_data, enc_time, enc_data, e_idx)
        self.mu = mu_updated
        return mu_updated

    def update(self, ranges, angles):
        '''
        Hook function to perform the particle weights update step of the Particle Filter SLAM
        '''
        updated_weights = self.observe.update(self.mu, self.weights, ranges, angles, self.num_particles)
        self.weights = updated_weights

    def add_texture(self, best_particle, cam_time, lidar_time):
        '''
        Hook function to update the texture map based on the stereo images
        '''
        t_map = self.texture.update_texture_map(best_particle, cam_time, lidar_time)
        #print(t_map)


if __name__ == "__main__":

    verbose = False
    l_params = lidar_params()
    f_params = fog_params()
    e_params = enc_params()
    c_params = stereo_cam_params()

    data_root = "../../data/"
    lidar_data_time, lidar_data = utils.read_data_from_csv(data_root+"sensor_data/lidar.csv")
    fog_data_time, fog_data = utils.read_data_from_csv(data_root+"sensor_data/fog.csv")
    enc_data_time, enc_data = utils.read_data_from_csv(data_root+"sensor_data/encoder.csv")  
    cam_time = utils.read_data_from_csv("../../cam_utils/cam_time.csv")
    if verbose:
        print("Lidar data for debug: ")
        lid = iter(zip(lidar_data_time, lidar_data[:, 0:2]))
        for i in range(5):
            print(lid.__next__())
        
        #Samples at rate 10 times more than lidar and encoder sampling rate
        print("FOG data for debug: ")
        fog = iter(zip(fog_data_time, fog_data[:, 0:2]))
        for i in range(5):
            print(fog.__next__())
        
        print("Encoder data for debug: ")
        enc = iter(zip(enc_data_time, enc_data))
        for i in range(5):
            print(enc.__next__())
        
        print(lidar_data_time.shape, lidar_data.shape)
        print(fog_data_time.shape, fog_data.shape)

    #Initialize Particle Filter SLAM class and SLAM away
    #Minor caveat: To reduce the run-time, I have considered every 5th lidar point. Resulting map is not much different than what it would
    #be if all the lidar data were used
    particle_filter = Particle_Filter_SLAM([lidar_data_time, lidar_data]) 
    particle_filter.SLAM([enc_data_time, enc_data], [lidar_data_time, lidar_data], [fog_data_time, fog_data[:,2]], cam_time[0])    
