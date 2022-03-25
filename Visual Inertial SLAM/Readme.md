
# Visual Inertial SLAM

### This folder contains source file for the Project 3 of ECE276A WI22. 

Note: Data for the project is included in the folder.

### High level organization:
|--code
|----/data
|-------/03.npz
|-------/10.npz
|----/main.py
|----/predict.py
|----/update.py
|----/pr3_utils.py
|----/requirements.txt

### Source Code Organization:
- code/predict.py - Contains prediction class used for EKF prediction step. Uses motion model to predict updated IMU pose.
- code/update.py - Contains update class. "ekf_update" function of this class is responsible for updating the IMU pose as well as the landmarks. 
- code/main.py - Entry point of the code base. Loops through the data provided for all time steps and makes appropriate calls to predict and update function.
- code/pr3_utils.py - Contains class and functions that are used for vizualization and reading data.

### Instructions for running the code:

- Install the libraries mentioned in the requirements.txt, a simple pip3 install -r requirements.txt will suffice
Part a) and b):
- Ensure that the variable vi_slam in main.py is set to False. "ekf_update" function of update.py relies on this flag to either run full SLAM or just the map update

Part c): Repeat the above step with vi_slam variable set to True.

Note: The default code considers only a subset of the entire feature set. Currently it is set to 200, to change the number of features, please update the "num_features_kept" varaible in main.py to desired value.
    More number of features takes more time. Depending on the RAM size, code execution gets really slow if the features kept are greater than 500.

Please note that I have added another variable to "visualize_trajectory_2d" to visualize the landmarks as it was missing in the default code. 

