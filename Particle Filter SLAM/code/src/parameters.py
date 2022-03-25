'''
Contains parameter classes of the sensors used in the project
These parameters are the ones provided along with the data
fog - Fibre Optic Gyro
enc - Encoder
'''

import numpy as np

class lidar_params():
    def __init__(self):
        self.FOV = 190
        self.s_angle = -1
        self.e_angle_max = 185
        self.e_angle_min = -5
        self.ang_res = 0.666
        self.max_range = 80

        self.lidar_rot = np.array([[0.00130201, 0.796097, 0.605167],
                                   [0.999999, -0.000419027, -0.00160026],
                                   [-0.00102038, 0.605169, -0.796097]], dtype=np.float32)

        self.lidar_pos = np.array([[0.8349, -0.0126869, 1.76416]], dtype=np.float32)
        self.lidar_rpy = np.array([[142.759, 0.0584636, 89.9254]], dtype=np.float32)
        rot = np.vstack((self.lidar_rot, np.zeros((1, 3))))
        pos = np.vstack((self.lidar_pos.T, 1))
        self.lidar_transformation = np.hstack((rot, pos))

    
class fog_params():
    def __init__(self):
        self.fog_rot = np.array([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]], dtype=np.float32)

        self.fog_pos = np.array([[-0.335, -0.035, 0.78]], dtype=np.float32)
        self.fog_rpy = np.array([[0, 0, 0]], dtype=np.float32)

class enc_params():
    def __init__(self):
        self.enc_res = 4096
        self.enc_l_diam = 0.623479
        self.enc_r_diam = 0.622806
        self.wheel_base = 1.52439

class stereo_cam_params():
    def __init__(self):
        self.stereo_rot = np.array([[-0.00680499, -0.0153215, 0.99985],
                                   [-0.999977, 0.000334627, -0.00680066],
                                   [-0.000230383, -0.999883, -0.0153234]], dtype=np.float32)

        self.stereo_pos = np.array([[1.64239, 0.247401, 1.58411]], dtype=np.float32)
        self.stereo_rpy = np.array([[-90.878, 0.0132, -90.3899]], dtype=np.float32)
        self.stereo_b_mm = 475.143600050775
        self.left_intrinsic = np.array([[7.7537235550066748e+02, 0., 6.1947309112548828e+02,],
                                        [0., 7.7537235550066748e+02, 2.5718049049377441e+02,],
                                        [0., 0., 1.]])
        self.R_o_r = np.array([[0., -1., 0],
                        [0., 0., -1.],
                        [1., 0., 0]], dtype=np.float64)

        rot = np.vstack((self.stereo_rot, np.zeros((1, 3))))
        pos = np.vstack((self.stereo_pos.T, 1))
        self.camera_T = np.hstack((rot, pos))

        #Values from the YAML file
        self.left_cu = 619.37
        self.left_cv = 257.18
        self.left_fs = 775.37

