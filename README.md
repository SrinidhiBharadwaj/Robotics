# Project Structure

- bin_detection
   - color_training_data (Contains npy files with data collected from roipoly function)
   - parameters (Contains npy files for classifier weights of logistic regression and parameters of Gaussian Naive Bayes)
   - roipoly
   - Train-predict.ipynb (File in which the model is trained)
   - bin_detector.py
   - data_generator.py (Script to collect and store data of different classes. Collected data is stored in /color_training_data
   - pixel_classifier.py (Python script that contains the classifier. This file is exactly same as the one used for problem 1, only difference being the number of classes here is 5 instead of 3, copied the file into this folder for ease of use and debugging)
   - test_bin_detection_score.py (Script that I used to check my classification output to finetune the model)
   - test_bin_detector.py (Clone of run_test.py)
   - test_roipoly.py
   
 - pixel_classification
    - parameters (Contains npy files for classifier weights of logistic regression and parameters of Gaussian Naive Bayes)
    - Train-LogReg.ipynb (File in which the model is trained)
    - pixel_classifier.py (File with core logic, contains model and prediction code)
    - test_pixel_classification_score.py (Script to test accuracy)
    - test_pixel_classifier.py
 
All the files provided here can be run independently. Default classifier is set to Logistic Regression in pixel_classifier.py, to test GaussianNaiveBayes, please replace self.clf = "LogisticRegression" with "GaussianNaiveBayes" in pixel_classifier.py files.


