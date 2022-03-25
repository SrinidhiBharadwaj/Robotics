'''
This file contains source code for the pose prediction step of VI_SLAM
Pose covariance noise values were arrived at empiracally by running the algorithm with multiple noise values
Output of the SLAM seems to be dependent on the noise values: More information in the report
'''

import numpy as np
import pr3_utils as utils
import scipy.linalg
class predict():
    def __init__(self, map):
        #Noise values were arrived at empiraclly after running multiple experiments
        self.W = np.block([[0.1*np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), 0.18000000000000002*np.eye(3)]])
        self.map = map

    def predict(self, tau, ang_vel, lin_vel, priors):
        '''
        Used by the main function in VI SLAM during prediction process
        Provides call to EKF predict function
        '''
        control_input = np.vstack((lin_vel.reshape(-1, 1), ang_vel.reshape(-1, 1)))

        u_hat, w_hat, v_hat = self.u_hat(lin_vel, ang_vel)
        u_hat_r6 = self.get_u_t_hat_r6(w_hat, v_hat)

        mu, cov = self.ekf_predict(tau, priors, u_hat, u_hat_r6)
        return mu, cov

    def ekf_predict(self, tau, priors, u_hat, u_hat_r6):
        '''
        Performs the EKF prediction step
        Inputs: 
        tau: Time elapsed 
        priors: Tuple containing pose_mu and pose_covariance
        u_hat: Hat matrix used for exponentiation
        u_hat_r6: Hat matrix for covariance update

        Note: Combined covariance for full slam is updated in main file's "localize_with_IMU" function after each 
        prediction step. Choice of this design is to keep code modular and rapid implementation of VI SLAM
        '''
        p_mu, p_cov = priors[0], priors[1]
        updated_mu =  p_mu @ scipy.linalg.expm(tau*u_hat)
        updated_cov = scipy.linalg.expm(-tau*u_hat_r6) @ p_cov @ scipy.linalg.expm(-tau*u_hat_r6).T + self.W
        return updated_mu, updated_cov
    
    def get_u_t_hat_r6(self, w_hat, v_hat):
        '''
        Helper function to return the matrix used for predicting new covariance values
        Returned function is in R^{6x6}
        '''
        u_hat_r6 = np.block([[w_hat, v_hat], [np.zeros((3, 3)), w_hat]])
        #print(u_hat_r6)
        return u_hat_r6

    def u_hat(self, lin_velocity, ang_velocity):
        '''
        Helper function that returns the hat matrix for control inputs
        Returned matrix is a block matrix comprising of angular velocity, linear velocity and zeros as below
        u_hat = [[w^, v]
                 [0.T, 0]]
        Equation on Slide 27 or Lecture 13
        '''
        w_hat = self.get_hat_matrix(ang_velocity)
        v_hat = self.get_hat_matrix(lin_velocity)
        u_hat = np.block([[w_hat, lin_velocity.reshape(-1, 1)], [np.zeros((1, 3)), 0]])
        #print(u_hat)
        return u_hat, w_hat, v_hat
    
    def get_hat_matrix(self, x):
        '''
        Returns the hat matrix according to the following equation
        Input = [x, y, z]
        Output = [[0, -z, y],
                  [z, 0, -x],
                  [-y, x, 0]]
        '''
        x = x.reshape(-1, 3)
        
        x_hat = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
        x_hat[1][0] = x[0][2]
        x_hat[2][0] = -x[0][1]

        x_hat[0][1] = -x[0][2]
        x_hat[2][1] = x[0][0]

        x_hat[0][2] = x[0][1]
        x_hat[1][2] = -x[0][0]
        
        return x_hat