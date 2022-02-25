'''
Module that is used to generate dataset for blue recycling bin detection
Used roipoly for pixel selection
Saves the selected pixel values in an npy file for later use in training
'''
import numpy as np
import os
import cv2
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import matplotlib
from itertools import compress
matplotlib.use('Qt5Agg')

def generate_data(path, color='random'):

    n = len(next(os.walk(path))[2]) # number of files
    X = np.empty([n, 3])
    i = 0
    colored_image = np.zeros((1, 3))

    for filename in os.listdir(path):  
        print("Labelling data for color {}..".format(color))
        print(i)
        i=i+1
        print(os.path.join(path,filename))
        img = cv2.imread(os.path.join(path, filename))

        #Convert BGR to HSV space to better detect colors
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
        my_roi = RoiPoly(fig=fig, ax=ax, color='r')

        # get the image mask
        mask = my_roi.get_mask(img)
        
        # display the labeled region and the image mask
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('%d pixels selected\n' % img[mask,:].shape[0])

        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
        ax1.add_line(plt.Line2D(my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
        ax2.imshow(mask)
        
        plt.show(block=True)
        #Save the numpy values to be later stored in a .npy file
        colored_image = np.vstack([colored_image, np.unique(img[mask], axis=0)])
        
    
    print("Storing the {} training data..")
    np.save("color_training_data/{}.npy".format(color), colored_image)

if __name__ == '__main__':
  folder = 'data/training'
  generate_data(folder, "bin-blue")
  
     
