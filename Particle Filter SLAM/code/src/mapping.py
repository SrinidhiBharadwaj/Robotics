'''
This file contains the class for core mapping logic using lidar scans
Map values are initialized in the constructor where the map size and 
resolution are predefined.

Map resolution is a hyperparameter.
'''
import numpy as np
from parameters import lidar_params
import parameters as params
import pr2_utils as utils

class Mapping():
    def __init__(self, lidar_data):
        self.m_xmax, self.m_xmin = 2000, -200
        self.m_ymax, self.m_ymin = 500, -1500
        self.m_res = 0.5 #Map resolution
        self.num_cells_x = int(np.ceil((self.m_xmax - self.m_xmin)/self.m_res + 1))
        self.num_cells_y = int(np.ceil((self.m_ymax - self.m_ymin)/self.m_res + 1))
        #Initialize the map with zeros(free)
        self.m = np.zeros((self.num_cells_x, self.num_cells_y), dtype=np.float32)
        self.t_map = np.zeros((self.num_cells_x, self.num_cells_y, 3), dtype=np.uint8) # Texture map
        self.m_exp =((1-1/(1+np.exp(self.m))) < 0.1).astype(np.int)
        self.l_params = lidar_params() 
        self.angles = np.arange(self.l_params.e_angle_min, self.l_params.e_angle_max, self.l_params.ang_res) * (np.pi/180)
        self.lidar_time = lidar_data[0]
        self.lidar_data = lidar_data[1]
        

    def update_map(self, particles, ranges, angles):
        '''
        Core function of mapping
        Converts the lidar scan to cartesian coordinates
        Converted values are transfromed to body frame and from body to the world frame
        Map cells hit by the laser are calculates using bresenham2D function and the log odds are updated
        Max and min values are between -10 and 10 for the map
        '''
        b_T_w = self.get_body_to_world_transform(particles[0], particles[1], particles[2])
        #Following slide 17 of lecture 10
        x, y = self.convert_scan_to_cartesian(ranges, angles)
        cartesian_coordinates = np.array(list(zip(x, y))).reshape(2, -1)
        cartesian_coordinates = np.vstack((cartesian_coordinates, np.ones((2, cartesian_coordinates.shape[1]))))
        cartesian_coordinates[2,:] = 0.5

        body_frame = self.convert_to_body(cartesian_coordinates)
        world_frame = b_T_w @ body_frame
        #print(particles.shape)

        sx = np.ceil((particles[0] - self.m_xmin) / self.m_res).astype(np.int32) - 1 
        sy = np.ceil((particles[1] - self.m_ymin) / self.m_res).astype(np.int32) - 1

        ex = np.ceil((world_frame[0,:] - self.m_xmin) / self.m_res).astype(np.int32) - 1 
        ey = np.ceil((world_frame[1,:] - self.m_ymin) / self.m_res).astype(np.int32) - 1

        for scan in range(np.size(ranges)):
            #Calculate cells that are hit by the laser
            occupied = utils.bresenham2D(sx, sy, ex[scan], ey[scan])
            #Take those indices that are present in the map
            valid_indices = np.logical_and(np.logical_and(np.logical_and((occupied[0, :] >= 0), (occupied[1, :] >= 0)), 
                                                (occupied[0, :] < self.num_cells_x)), (occupied[1, :] < self.num_cells_y))
            
            self.update_map_log_odds(occupied, valid_indices, ex[scan], ey[scan])
            
        
        log_odds_max = 10 * np.log(4)
        log_odds_min = -10 * np.log(4)
        self.m = np.clip(self.m, log_odds_min, log_odds_max)
        #print(np.min(self.m), np.max(self.m))
        updated_map_debug = np.copy(self.m)
        return updated_map_debug
    
    def update_map_log_odds(self, cells, indices, endx_val, endy_val):
        '''
        Map log odds are updates based on the output of the bresenham2D function
        Occupied map cells are added log(4) and empty cells are subtracted with log(4)
        
        '''
        #print(indices)
        x, y = cells[0,:].astype(np.int16), cells[1,:].astype(np.int16)
        self.m[x[indices], y[indices]] -= np.log(4) #4 is a hyperparameter

        if ((endx_val >= 0) and (endx_val < self.num_cells_x) and 
                    (endy_val >= 0) and (endy_val < self.num_cells_y)):
            self.m[endx_val,endy_val] += np.log(4)

    def get_valid_lidar_ranges(self, l_idx):
        '''
        Helper function to return lidar ranges between 0.3 and 40
        '''
        l_ranges = self.lidar_data[l_idx, :]
        valid_indices = np.logical_and((l_ranges<40), (l_ranges>0.3))
        l_ranges = l_ranges[valid_indices]
        return l_ranges, self.angles[valid_indices]

    def convert_scan_to_cartesian(self, ranges, angles):
        '''
        Helper function to convert scan values to cartesian coordinates
        '''
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        return x.reshape(1, x.shape[0]), y.reshape(1, y.shape[0])

    def convert_to_body(self, points):
        '''
        Transforming points from lidar to body frame
        Transformation matrix is already provided
        '''
        b_T_l = self.l_params.lidar_transformation
        return b_T_l @ points  

    def get_body_to_world_transform(self, x, y, theta):
        '''
        Helper function to provide the transformation matrices based on the x, y and theta values
        Used extensively to convert points from body to world frame
        Ex: Lidar points are converted from sensor to body frame and from body to world frame 
        which is later used to update the map
        '''
        a = np.cos(theta)
        b = np.sin(theta)
        w_R_b = np.array([[a, b, 0],
                          [-b, a, 0],
                          [0, 0, 1]], dtype=np.float32)
        w_p_b = np.array([[x, y, 0]], dtype=np.float32) #Setting z-coordinate to 0
        w_T_b = np.hstack((np.vstack((w_R_b, np.zeros((1, 3)))), np.vstack((w_p_b.T, 1))))
        return w_T_b