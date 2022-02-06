'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import cv2
from skimage.measure import label, regionprops
import os


class BinDetector():
    folder_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(folder_path, 'parameters/classifier_weights.npy')
    mean = os.path.join(folder_path, 'parameters/pixel_mean.npy')
    sigma = os.path.join(folder_path, 'parameters/pixel_var.npy')
    theta = os.path.join(folder_path, 'parameters/pixel_theta.npy')

    def __init__(self):
        '''
                Initilize your bin detector with the attributes you need,
                e.g., parameters of your classifier
        '''
        self.weights = np.load(BinDetector.model_path)
        self.pixel_mean = np.load(BinDetector.mean)
        self.pixel_var = np.load(BinDetector.sigma)
        self.pixel_prior = np.load(BinDetector.theta)
        self.color_classes = [1, 2, 3, 4, 5]
        self.clf_name = "LogisticRegression"

    def softmax(self, z):
        val = np.exp(z)/np.sum(np.exp(z), axis=1).reshape(-1, 1)
        return val

    # Predict the mask based on the weigths of the classifier
    def predictLogistic(self, X):
        X = np.insert(X, 0, 1, axis=1)
        # Logistic regression to calculate the output and create masks
        z = np.dot(X, self.weights.T).reshape(-1, len(self.color_classes))
        self.probabilities = self.softmax(z)
        # Pick the class with highest probability
        # Vectorized method to save time for dataset with many pixels
        y = np.vectorize(lambda c: self.color_classes[c])(
            np.argmax(self.probabilities, axis=1))
        return y

    def predictGaussian(self, X):
        y = np.zeros((X.shape[0], 1))
        posterior_prob = np.zeros((len(self.color_classes), 1))
        # Using the learned parameters
        for row in range(X.shape[0]):
            for cls in range(len(self.color_classes)):
                prob_class_color = 0
                for col in range(X.shape[1]):
                    prob_class_color += np.log(self.pixel_var[cls, col]) + (((X[row, col]
                                                                              - self.pixel_mean[cls, col])**2)/self.pixel_var[cls, col])
                    #print(prob_class_color)
                posterior_prob[cls] = prob_class_color + \
                    np.log(1/self.pixel_prior[cls]**2)
            y[row] = np.argmin(posterior_prob) + 1
        return y

    def recursive_erosion(self, img):
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)

        image_area = img.shape[0] * img.shape[1]
        contours, hierarchy = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        boxes = []
        # print(image_area/200)
        for i in range(np.shape(contours)[0]):
            # print(cv2.contourArea(contours[i]))
            if (cv2.contourArea(contours[i]) > image_area/200):
                x, y, lengthX, lengthY = cv2.boundingRect(contours[i])
                if lengthY < 2.5 * lengthX and lengthY > 1 * lengthX:
                    boxes.append([x, y, x + lengthX, y + lengthY])
                    break
                else:
                    boxes.append(self.recursive_erosion(img))
        return boxes
               
    def segment_image(self, img):
        '''
                Obtain a segmented image using a color classifier,
                e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
                call other functions in this class if needed

                Inputs:
                        img - original image
                Outputs:
                        mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE
        # Convert to HSV color space as the logistic regression model is trained on HSV color space
        # It is assumed here that that image input is in BGR space
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rows, cols, channels = img.shape

        img_hsv = img_hsv.astype(np.float64)/255
        img_hsv = np.reshape(img_hsv, (rows*cols, channels))
        # Obtain the prediction mask
        if self.clf_name == "LogisticRegression":
            y = self.predictLogistic(img_hsv)
        else:
            y = self.predictGaussian(img_hsv)

        # For debugging purpose
        mask = y
        mask[mask == 1] = 255
        mask[mask == 2] = 128
        mask[mask == 3] = 75
        mask[mask == 4] = 40
        mask[mask == 5] = 0

        # Reshape to original size
        mask_img = mask.reshape(rows, cols)
        mask_img = mask_img.astype(np.uint8)

        # YOUR CODE BEFORE THIS LINE
        ################################################################
        return mask_img

    def get_bounding_boxes(self, img):
        '''
                Find the bounding boxes of the recycling bins
                call other functions in this class if needed

                Inputs:
                        img - original image
                Outputs:
                        boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
                        where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE

        # Set all the other elements other than blue to be 0
        img[img != 255] = 0
        kernel = np.ones((12, 12), np.uint8)

        image_area = img.shape[0] * img.shape[1]
         # Obtain contours
        contours, _ = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        boxes = []
        nd_contours = np.asarray(contours, dtype=object)
        num_contours = nd_contours.shape[0]
        for i in range(num_contours):
            # print(cv2.contourArea(contours[i]))
            if (cv2.contourArea(contours[i]) > image_area/200):
                x, y, width, height = cv2.boundingRect(contours[i])
                # Bin statistics (not scalable for other items)
                if height < 2.5 * width and height > 1 * width:
                    boxes.append([x, y, x + width, y + height])

        #boxes = self.recursive_erosion(img)

        # YOUR CODE BEFORE THIS LINE
        ################################################################

        return boxes
