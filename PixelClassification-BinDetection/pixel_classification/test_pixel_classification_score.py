import unittest
from pixel_classifier import PixelClassifier
import generate_rgb_data
import cv2, os, glob
import numpy as np 

"""Pixel Classifier"""
img_folder = "../tests/testset/pixel_classification"
color_dict = {1:"/red", 2:"/green", 3:"/blue"}
score = 0
myPixelClassifier = PixelClassifier()
for c in range(len(color_dict)):
    folder = img_folder+str(color_dict[c+1])
    X = generate_rgb_data.read_pixels(folder)
    y = myPixelClassifier.classify(X)
    print((sum(y==(c+1))/y.shape[0]) * 10 / len(color_dict))
    score += (sum(y==(c+1))/y.shape[0]) * 10 / len(color_dict)
print(score)