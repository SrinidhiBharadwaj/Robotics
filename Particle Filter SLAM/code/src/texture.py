'''
This file contains the texture class which is used to update the texture of the predicted map based 
on the RGB values.
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
from parameters import stereo_cam_params
import itertools
import pr2_utils as utils

class Texture():
    def __init__(self, map):
        self.data_root = "../../data/stereo_images/"
        self.cam_params = stereo_cam_params()
        self.map_res = map.m_res
        self.xmin, self.xmax = map.m_xmin, map.m_xmax
        self.ymin, self.ymax = map.m_ymin, map.m_ymax
        self.map_ins = map


    def update_texture_map(self, particle, cam_time, l_time):
        '''
        This function updates the texture map based on the camera observation at a given time
        Depth map is estimated from the stereo images whose world coordinates are calculated
        Based on the calculated world coordinates, the RGB value is assigned from the left stereo image
        '''
        x_t_k, y_t_k, theta_t_k = particle[0], particle[1], particle[2]
        #Obtain the transformation matrix based on the particle values at time t
        w_T_b = self.map_ins.get_body_to_world_transform(x_t_k, y_t_k, theta_t_k)
        
        #Create disparity image
        image_name = self.get_image_name(cam_time, l_time)
        image_l, disparity = self.read_images(image_name)
        depth_pts, pix_points = self.get_points_on_left_image(disparity)
        #print(depth_pts)

        #Convert points from robot body to world coordinates
        pixel_world = w_T_b @ depth_pts.T
        pixel_world = pixel_world[:2]
       
        #print(depth_pts.shape)
        mx = np.ceil((pixel_world[0,: ]  - self.map_ins.m_xmin) /self.map_ins.m_res).astype(np.int16) - 1
        my = np.ceil((pixel_world[1, :]  - self.map_ins.m_ymin) /self.map_ins.m_res).astype(np.int16) - 1
        #print(np.unique(mx))
        
        #Update the texture map with RGB values from the left stereo image and return(for debugging)
        self.map_ins.t_map[mx,my,:] = image_l[pix_points]
        return np.copy(self.map_ins.t_map)
	   
        
    def get_points_on_left_image(self, disparity):
        '''
        Uses the disparity image created using right and left stereo images using stereoBM function
        to calculate the real world coordinates
        
        Z = Bf/disparity  #Small value is added to dispartiy to avoid NaN values
        cv2.rgb.depthTo3d returns the real world coordinates

        Return:
        Depth points calculated
        Corresponding indices which are used to pick the RGB values for the texture map
        '''

        Z = self.cam_params.left_fs * self.cam_params.stereo_b_mm * 1e-3 / (disparity + 1e-6)
        
        Z[Z > 50] = 0.0
        #print(depth_img)
        left_depth_image = Z.copy()
        optical_coord = cv2.rgbd.depthTo3d(left_depth_image.astype(np.float32), self.cam_params.left_intrinsic)
        homog_coord = np.ones((optical_coord.shape[0], optical_coord.shape[1], 4))
        homog_coord[..., :3] = optical_coord.copy()
        homog_coord = homog_coord.transpose(2,0,1)

        robot_frame = np.einsum("ij, jkl -> ikl", self.cam_params.camera_T, homog_coord).transpose(1,2,0)
        good_pts_idx = np.where((robot_frame[...,2] > 0.2))
        depth_pts = robot_frame[good_pts_idx]
        
        return depth_pts, good_pts_idx

    def get_image_name(self, cam_time, l_time):
        '''
        Helper function to return the stereo image names based on lidar and camera timestamp
        '''
        #Image names and lidar values are not synced perfectly, need to compare
        #Similar logic as picking FOG data
        cam_idx = np.asarray(np.abs(cam_time - l_time).argmin())
        image_name = str(cam_time[cam_idx]) + ".png"
        return image_name

    def read_images(self,image_name):
        '''
        Helper function to read and return images based on the image name
        '''
        left_path = self.data_root+"stereo_left/"+image_name
        image_l = cv2.imread(left_path, 0)
        image_l = cv2.cvtColor(image_l, cv2.COLOR_BAYER_BG2BGR)

        disparity_path = self.data_root+"disparity/"+image_name
        disparity = cv2.imread(disparity_path)
        disparity = cv2.cvtColor(disparity, cv2.COLOR_BGR2GRAY)

        return image_l, disparity

    def get_real_world_coord(self, disparity):
        '''
        Unoptimized function for obtaining the real world coordinates
        An optimized function is prepare_stereo_pts()
        '''
        print(disparity.shape)
        z = self.cam_params.left_fs * self.cam_params.stereo_b_mm * 1e-3 / (disparity + 1e-6)
        
        z[z>50] = 0 #Filter out values greater than mts
        # z_max = np.max(z)
        # z[z_max]
        intrinsic = self.cam_params.left_intrinsic
        fsub_row = np.array([0, 0, 0, self.cam_params.left_fs * self.cam_params.stereo_b_mm * 1e-3])
        #intrinsic = np.hstack((intrinsic, np.zeros((3, 1))))
        intrinsic = np.insert(intrinsic, 2, fsub_row, axis=0) #(4x4)

        #print(np.min(z))
        world_coord = np.zeros((3, np.size(disparity)))
        i = 0
        #Get X, Y values
        for v, u in itertools.product(range(disparity.shape[0]), range(disparity.shape[1])):
            pixel_coord = np.array([v, u, disparity[v, u]]).reshape(3, 1)
            pixel_coord = np.vstack((pixel_coord, 1)) #1x4, homogeneous tranformation
            #print(pixel_coord)
            optical_coord = np.linalg.inv(intrinsic) @ pixel_coord #4x1 matrix
            optical_coord = z[v, u]*optical_coord
            optical_coord = np.delete(optical_coord, 3, axis=0)
            #print(optical_coord)
            world_translated_coord = self.cam_params.stereo_rot @ optical_coord
            world_coord_pixels = world_translated_coord + self.cam_params.stereo_pos.T  
            world_coord[:, i] = np.copy(world_coord_pixels[:1])
            i+=1
        
        return world_coord


    
        

        