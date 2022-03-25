'''
This file contains class for the prediction step of the Particle filter
using differential drive motion model

'''

import numpy as np
from parameters import enc_params, fog_params

class Prediction():
    def __init__(self):
        self.enc = enc_params()
        self.fog = fog_params()
        self.ang_vel = 0

    def motion_model(self, mu, fog_time, fog_data, enc_time, enc_data, e_idx=1):
        '''
        Function to predict the state of the robot at (t+1) given the state at t
        mu -> Current State
        fog -> Control input for yaw
        encoder -> Control input for x and y
        
        Return: Updated mu ([x_(t+1), y_(t+1), theta_(t+1)])
        '''

        #Converting to seconds from nanoseconds
        enc_time = enc_time * 10e-9
        fog_time = fog_time * 10e-9

        temp_mu = mu.copy()

        i = e_idx
        #Time calculation
        delta_t = enc_time[i] - enc_time[i-1]
        #Velocity calculation
        l_enc = enc_data[i][0] - enc_data[i-1][0]
        r_enc = enc_data[i][1] - enc_data[i-1][1]

        vl = (np.pi * self.enc.enc_l_diam * l_enc) / (self.enc.enc_res * delta_t)
        vr = (np.pi * self.enc.enc_r_diam * r_enc) / (self.enc.enc_res * delta_t)
        v = (vl + vr)/2
        fog_idx = np.asarray(np.abs(enc_time[i] - fog_time).argmin())
        
        if fog_idx < (np.size(fog_time) - 10):
            self.ang_vel = (np.asarray([0, 0, np.sum(fog_data[fog_idx:fog_idx+10])/(fog_time[fog_idx+10] - fog_time[fog_idx])]))[-1]
        else:
            self.ang_vel = self.ang_vel

        #Obtain the prediction value and return
        x, y, theta = self.predict(temp_mu, v, self.ang_vel, delta_t, add_noise=True)
        temp_mu[0,:] = x
        temp_mu[1,:] = y
        temp_mu[2,:] = theta

        return temp_mu
    
    def predict(self, mu, v, ang_vel, delta_t, add_noise=False):
        '''
        Predicts the state of the particles stored in mu based on the motion model as below
        mu = [x, y, z].T (For N particles)
        Motion model:
        x_(t+1) = x + v * t * cos(theta) + Noise
        y_(t+1) = y + v * t * sin(theta) + Noise
        theta_(t+1) = theta + t * angular_velocity + Noise

        Noise covariance is in the order of 1e-4 and mean is 0
        '''
        theta = mu[2, :] #All thetas 
        x = mu[0,: ] + v * delta_t * np.cos(theta)
        y = mu[1,: ] + v * delta_t * np.sin(theta)
        theta = theta + ang_vel * delta_t
        #Royally adding wrong amount of noise! Setting flag to false for now
        if add_noise:
            #print([np.random.normal(0, np.abs(np.max(v * delta_t * np.cos(theta)))/10 , mu.shape[1])])
            x = x + np.array([np.random.normal(0, np.abs(np.max(v * delta_t * np.cos(theta)))/10, mu.shape[1])])
            y = y + np.array([np.random.normal(0, np.abs(np.max(v * delta_t * np.sin(theta)))/10, mu.shape[1])])
            theta = theta + np.array([np.random.normal(0, np.abs(np.max(ang_vel * delta_t))/10, mu.shape[1])])
        return x, y, theta

    def motion_model_vanilla(self, mu, fog_time, fog_data, enc_time, enc_data, e_idx=1):
        '''
        Unused function
        Added this to test out the model before vectorizing it
        '''
        
        traj_x, traj_y = [], []
        enc_time = enc_time * 10e-9
        fog_time = fog_time * 10e-9

        for i in range(1, np.size(enc_time)):
            delta_t = enc_time[i] - enc_time[i-1]

            l_enc = enc_data[i][0] - enc_data[i-1][0]
            r_enc = enc_data[i][1] - enc_data[i-1][1]

            vl = (np.pi * self.enc.enc_l_diam * l_enc) / (self.enc.enc_res * delta_t)
            vr = (np.pi * self.enc.enc_r_diam * r_enc) / (self.enc.enc_res * delta_t)
            v = (vl + vr)/2
            fog_idx = np.asarray(np.abs(enc_time[i] - fog_time).argmin())
            
            if fog_idx < (np.size(fog_time) -10):
                ang_vel = (np.asarray([0, 0, np.sum(fog_data[fog_idx:fog_idx+10])/(fog_time[fog_idx+10] - fog_time[fog_idx])]))[-1]
            else:
                ang_vel = ang_vel

            for j in range(1):
                x, y, theta = self.predict(mu, v, ang_vel, delta_t)
                mu = np.array([x, y, theta])
                traj_x.append(x)
                traj_y.append(y)

        return traj_x, traj_y