'''
This file contains code for the EKF update step for both mapping and localization
Utilizes the camera observation model to update the values
Dimensions of the matrices used in the file are based on the lecture notes

Note: Function "ekf_update" contains a switch "vi_slam" which, if set to False just runs map update
and if set to True, runs full VI_SLAM.
'''
import numpy as np
import scipy
import pr3_utils as utils

class update():
    def __init__(self, k, iTc, b, features):
        self.cam_intrinsic = k 
        self.iTc = iTc #c = Camera, i = IMU
        self.cTi = np.linalg.inv(self.iTc)
        self.baseline = b 
        self.features = features
        fsub = self.cam_intrinsic[0, 0] * self.baseline
        fsub_row = np.array([[0, 0, 0, fsub]], dtype=np.float32)
        intrinsic = np.hstack((self.cam_intrinsic, np.zeros((3, 1))))
        self.intrinsic = np.insert(intrinsic, 2, fsub_row, axis=0)
        self.M = np.array([[k[0, 0], k[0, 1], k[0, 2], 0],
                           [k[1, 0], k[1, 1], k[1, 2], 0],
                           [k[0, 0], k[0, 1], k[0, 2], -fsub],
                           [k[1, 0], k[1, 1], k[1, 2], 0]], dtype=np.float64)
        self.num_landmarks_seen = 0
        self.world_to_imu = 0
        #Below matrix is used for H matrix calculation
        self.P = np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 0.]], dtype=np.float64)

        self.obs_noise = 100 * np.eye(4) #50 is a hyperparameter, size 4 as z_t is a 4x1 vector
        self.updated_indices = np.empty(0,dtype=int) #Flag used to check if I have observed a given index
        self.valid_indices = 0 #Not the best SW engineering principle
        self.imu_cov = 0 #Initialized to dummy values
        self.imu_mu = 0 #Initialized to dummy values

    def update(self, prior_mu, prior_cov, imu_mu_inv, imu_cov, combined_cov, time_idx=0, vi_slam=False):
        '''
        Hook function that prepares the data and passes it on to the ekf update step
        It also adds newly observed landmarks into the prior matrix
        '''
        #self.world_to_imu is iTw: Transformation matrix from world to IMU
        self.world_to_imu = imu_mu_inv
        
        #Inverting because I am passing the inverse of predicted value. self.imu_mu is the pose provided by the prediction step
        #This variable gets updated during VI SLAM update. (Inversion is done to avoid changing my code for previous parts)
        self.imu_mu = np.linalg.inv(imu_mu_inv)
        self.imu_cov = imu_cov
        features_t = self.features[:,:,time_idx]
        valid_indices, _ = self.get_valid_landmarks(features_t) #Get all the columns with -1s

        #Get the indices where the values are 0s
        indices_to_update = np.where(np.sum(prior_mu, axis=1) == 0)

        #Get the indices that are newly seen at time 't' -> Indices that I am seeing now and 
        #the indices that I have not seen before
        indices_to_update = np.intersect1d(indices_to_update, valid_indices)
        #print(f"valid_indices{valid_indices}, indices to update {indices_to_update}")

        #Get the indices that are neeeded to be updated at time 't' -> Indices that I have seen before and the indices
        #that I am seeing now
        indices_observed = np.intersect1d(valid_indices, self.updated_indices)

        if (np.size(indices_observed) != 0):
            #Update the number of landmarks seen so far, not used in the current version of the code
            self.num_landmarks_seen = max(self.num_landmarks_seen, valid_indices.max()+1)       
            prior_mu, combined_cov, prior_cov = self.ekf_update(prior_mu, prior_cov, features_t, indices_observed, combined_cov, vi_slam=vi_slam)

        #Add the world coordinate points to indices that are being observed for the first time 
        u_L = features_t[0, indices_to_update]
        v_L = features_t[1, indices_to_update]
        u_R = features_t[2, indices_to_update]
        disparity = u_L - u_R
        pix_coord = np.vstack((u_L, v_L, disparity, np.ones(disparity.shape[0])))
        cam_coord = np.linalg.inv(self.intrinsic) @ pix_coord #4x1 matrix
        z = self.intrinsic[2, 3]/disparity
        cam_coord = z*cam_coord
        #Given transformation matrix is from imu to camera frame

        world_cord = np.linalg.inv(imu_mu_inv) @ self.iTc @ cam_coord

        prior_mu[indices_to_update, :] = world_cord[:3, :].T
        z = np.append(self.updated_indices, indices_to_update)
        self.updated_indices = np.unique(z)

        return prior_mu, combined_cov, prior_cov

    def ekf_update(self, p_mu, p_cov, features_t, valid_indices, combined_cov, vi_slam=True):
        '''
        Update function of the Extended Kalman Filter
        Equations follow the equations in slide 9 - Lecture 13
        
        Inputs:
        p_mu =          landmark piror mu
        p_cov =         landmark prior covariance
        features_t =    Features at time index t
        valid_indices = Indices of the landmarks that are seen before and are being observed currently
        combined_cov =  Coviarance matrix with both landmark covariance and pose covariance of size 3M+6 x 3M+6
        vi_slam =       Switch variable to either enable or disable SLAM
        '''
        #Obtain the Jacobians
        H_obs, H_pose, H_combined = self.get_Jacobian_H(p_mu, valid_indices)

        if vi_slam: #Part b and c
            #Calc Kalman gain
            k_gain = combined_cov @ H_combined.T @ np.linalg.inv(H_combined @ combined_cov @ H_combined.T + np.kron(self.obs_noise, np.eye(np.size(valid_indices))))
            z_tilde = self.get_observations(p_mu, valid_indices)
            z_t = features_t[:, valid_indices]    
            diff = (z_t - z_tilde).reshape(-1, 1, order='F')

            #Update landmark/map mu
            p_mu += (k_gain[:-6, :] @ diff).reshape(-1, 3)

            #Update pose mu based on the ekf update equation
            exponent = k_gain[:-6, :] @ diff
            exp_hat, _, _ = self.u_hat(exponent[:3, :], exponent[3:, :])
            self.imu_mu =  self.imu_mu @ scipy.linalg.expm(exp_hat)# (k_gain @ diff).reshape((-1, 3), order='F') 
            
            #Update combined covairance
            #Pose covariance is update in the main file's "landmark_mapping" function
            combined_cov= (np.eye(k_gain.shape[0]) - k_gain @ H_combined) @ combined_cov
           
        else: #If only landmark mapping - part b)
            #p_cov is landmark covariance matrix of size 3M x 3M
            sigma_t = p_cov
            k_gain = sigma_t @ H_obs.T @ np.linalg.inv((H_obs @ sigma_t @ H_obs.T + np.kron(self.obs_noise, np.eye(np.size(valid_indices)))))

            #Calc mu
            z_tilde = self.get_observations(p_mu, valid_indices)
            z_t = features_t[:, valid_indices]    
            diff = (z_t - z_tilde).reshape(-1, 1, order='F')

            #Calculate landmark_mu
            p_mu += (k_gain @ diff).reshape((-1, 3))
            
            #Calc $\sum_{t+1}$
            p_cov = (np.eye(k_gain.shape[0]) - k_gain @ H_obs) @ sigma_t 

        return p_mu, combined_cov, p_cov

    def is_invertible(self, a):
        # print(a.shape)
        # print(np.linalg.matrix_rank(a))
        return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

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
        return u_hat, w_hat, v_hat

    def get_observations(self, mu, indices):
        '''
        This function converts the calulates mu from IMU to camera co-ordinates
        Output of the function is used during mu update step
        '''
        #mu is a 3xNt vector, world co-ordinates   
        #Mu is in world, convert to imu and then to cam
        _mu = np.hstack((mu[indices, :], np.ones((np.size(indices), 1))))
       
        optical_coord = self.cTi  @ self.world_to_imu @ _mu.T
        optical_coord = optical_coord / optical_coord[2, :] #(Divison by z)
        z_tilde = self.M @ optical_coord #Provides uL, vL, uR, vR

        return z_tilde

    def get_Jacobian_H(self, prior, indices):  
        '''
        This function is used to calulate the Jacobian of the observation model
        Size of the H matrix is 4Nt x 3M, where Nt is the observed features in until
        time t and M is the total number of landmarks
        '''
        self.valid_indices = indices
        #Number features observed at time t
        num_valid_obs = np.size(indices)

        #H observation matrix is of size 4Nt x 3M
        rows = 4 * num_valid_obs #4N_t
        cols = 3 * prior.shape[0]  #3M

        H_obs = np.zeros((rows, cols), dtype=np.float64)
        H_pose = np.zeros((rows, 6), dtype=np.float64)
        H_combined = np.zeros((rows, cols+6), dtype=np.float64) # 4Nt x 3M+6

        valid_features = prior[indices, :]
        homo_valid = np.hstack((valid_features, np.ones((num_valid_obs, 1))))
        
        #H is a block matrix, updating values of each block for all the observations at time t
        for i in range(num_valid_obs):

            row_start = (i * 4) #In steps of 4
            row_end = row_start + 4#numpy slices do not consider the last index hence the +1
            col_start = indices[i] * 3
            col_end = col_start + 3 #going in steps of 3
            #Update block matrix for each observation
            #Converting x, y, z in world to IMU to camera
            cam_coord = self.cTi @ self.world_to_imu @ homo_valid[i, :].reshape(-1, 1)
            #Self note: Keep an eye out on the dimensions
            H_obs[row_start:row_end, col_start:col_end] =  self.M @ self.get_projfn_derivative(cam_coord) @ \
                                                        self.cTi @ self.world_to_imu @ self.P.T

            circle_dot = self.get_circle_dot((self.world_to_imu @ homo_valid[i, :].reshape(-1, 1))[:3]) #4N x 6
       
            H_pose[row_start:row_end, :] = self.M @ self.get_projfn_derivative(cam_coord) @ \
                                                        self.cTi @ circle_dot
            

            H_combined[row_start:row_end, col_start:col_end] = H_obs[row_start:row_end, col_start:col_end]
            H_combined[row_start:row_end, -6:] = H_pose[row_start:row_end, :]

        return H_obs, H_pose, H_combined

    def get_circle_dot(self, s):
        '''
        For the lack of better name, I am naming it circle_dot
        Function returns the "dot" matrix of s according to the following equation
        s_dot = [[I, -s_hat],
                 [0,  0]]
        size of s_dot: 4 x 6
        '''
        s_hat = self.get_hat_matrix(s) 
        s_hat = np.vstack((-s_hat, np.zeros((1, 3))))
        first_block = np.vstack((np.eye(3), np.zeros((1, 3))))
        ret_matrix = np.block([first_block, s_hat])
        
        return ret_matrix

    def get_projfn_derivative(self, q):
        '''
        q is the input vector in R4
        '''
        q_x = q[0]/q[2]
        q_y = q[1]/q[2]
        q_hom = q[3]/q[2]
        derivative = np.array([[1, 0, -q_x, 0],
                               [0, 1, -q_y, 0],
                               [0, 0, 0, 0],
                               [0, 0, -q_hom, 1]], dtype=np.float64)
        return (1 / q[2]) * derivative

    def get_valid_landmarks(self, features):
        '''
        Function to calculate the indices of the landmarks observed in the 
        image at time t
        Returns the indices and the feature values observed in the given image
        '''
        valid_indices = np.where((np.sum(features[:,:], axis=0) != -4))  
        valid_features = features[:,valid_indices].reshape(4, np.size(valid_indices))
     
        return valid_indices[0], valid_features

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
