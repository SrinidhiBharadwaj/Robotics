'''
This is the main file for PR3 for ECE276A WI22.
It contains the source code for VI SLAM on the given data

Visual_Intertial_SLAM's instance is used in the main function to provide calls to
EKF predict and update step

Embedded comments hopefully guide the locations to implementation of parts a, b and c
Parts b and c are implemented in the same function with a clear switch to differentiate between the two
if: vi_slam = False, part b
else: part c

Author: Srinidhi Kalgundi Srinivas
'''

import numpy as np
from pr3_utils import *
from ekf_predict import predict
from ekf_update import update
np.set_printoptions(suppress=True)
from tqdm import tqdm
import pickle

class Visual_Intertial_SLAM():
	def __init__(self, t, features, lin_velocity, ang_velocity, K, baseline, iTc):
		self.time = t
		self.features = features
		self.linear_velocity = lin_velocity
		self.angular_velocity = ang_velocity
		self.cam_intrinsic = K
		self.baseline = baseline
		self.iTc = iTc
		self.mapping = update(self.cam_intrinsic, self.iTc, self.baseline, self.features)
		self.localization = predict(self.mapping)
		
		self.num_landmarks = self.features.shape[1]
		self.imu_mu = np.eye(4, dtype=np.float64) #In SE(3) # Read as pose_mu
		self.imu_cov = np.eye(6, dtype=np.float64) #3Mx6 # Read as pose_cov

		self.combined_cov = np.eye(3*np.shape(features)[1] + 6, dtype=np.float64) #3Mx6
		self.landmark_mu =  np.zeros((self.num_landmarks, 3), dtype=np.float64)
		self.landmark_cov = np.identity(3*np.shape(features)[1])
		self.trajectory = np.zeros((4,4,np.size(self.time)), dtype=np.float64) 

	def predict(self, tau, time_idx):
		'''
		Hook function for pose prediction - Part a
		Provides call to EKF prediction function 
		'''
		self.imu_mu, self.imu_cov = self.localization.predict(tau, self.angular_velocity[: ,time_idx], self.linear_velocity[:, time_idx],
																(self.imu_mu, self.imu_cov))
		#For VI SLAM, the combined covariance's pose covariance part is updated with predicted pose covariance
		self.combined_cov[-6:, -6:] = self.imu_cov
		self.trajectory[:, :, time_idx] = (self.imu_mu)

	def update(self, time_idx, vi_slam):
		'''
		Hook functions for parts b and c
		Provides call to the update function which performs EKF update step
		'''
		self.landmark_mu, combined_cov, self.landmark_cov = self.mapping.update(self.landmark_mu, self.landmark_cov, np.linalg.inv(self.imu_mu), self.imu_cov, self.combined_cov, time_idx=time_idx,
																	vi_slam=vi_slam)

		# Below lines are for VI SLAM (part c), it does not affect part a or b as I am only updating the RHS only for SLAM
		# For part a and b, both LHS and RHS are the same, so assignment does not make any difference
		# Wrote it this way to avoid multiple if statements
		self.combined_cov = combined_cov
		self.imu_mu = self.mapping.imu_mu		
		self.imu_cov = self.combined_cov[-6:, -6:]

	def run_slam(self):
		'''
		Main function that starts the SLAM process. By default, part a) and c) are set to run(c in turn runs b)
		Please set vi_slam=False to run just part b. 
		Code for part c and b are seggregated in "update.py" file "ekf_update" function under a if else condition.
		'''

		vi_slam = True #Runs both part b and c when set to True
		time_idx = self.time.shape[1]
		print("Running Visual Intertial SLAM on the given data!!")

		#At t = 0, only update is performed (Observed values are inserted to their corresponding prior locations)
		self.update(0, vi_slam=vi_slam)
		for t in tqdm(range(1, time_idx)):
			# (a) IMU Localization via EKF Prediction
			tau = self.time[0, t] - self.time[0, t-1]
			self.predict(tau, t)

			# (b) Landmark Mapping via EKF Update
			#self.update(t, vi_slam=vi_slam)

			# (b) and (c) IMU value update based on the observed value (VI SLAM)
			# To just run part b, run this line with vi_slam value set to False
			self.update(t, vi_slam=vi_slam)

if __name__ == '__main__':

	# Load the measurements
	# Default dataset set to 03.npz. To reproduce the results mentioned in the report for 10, change the below line and rerun
	filename = "./data/03.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
	number_features_kept = 200
	num_features = features.shape[1] // number_features_kept

	#Algorithm is run for every 25th feature to avoid computational barriers without loss of generality and without hampering the results
	# (a), (b) and (c)
	vi_slam = Visual_Intertial_SLAM(t[:, :,], features[:, ::num_features, :, ], linear_velocity[:, :], angular_velocity[:,  :],K,b,imu_T_cam)	
	vi_slam.run_slam()				
	
	with open("slam_traj.pkl", 'wb') as f:
			pickle.dump([vi_slam.trajectory, vi_slam.landmark_mu ] , f)

	#Provided visualize trajectory code is modified to include plotting of mapped landmarks as well
	visualize_trajectory_2d(vi_slam.trajectory, vi_slam.landmark_mu, show_ori = True)


