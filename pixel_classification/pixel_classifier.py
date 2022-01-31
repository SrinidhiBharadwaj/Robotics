'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
import os


class PixelClassifier():
    def __init__(self):
        '''
                Initilize your classifier with any parameters and attributes you need
        '''
        #self.clf = GaussianNaiveBayes(num_classes=3)
        self.clf = LogisticRegression()

    def train(self, X, y, lr=0.001, batch_size=32, epochs=100, verbose=False):
        self.clf.train(X, y, lr, batch_size, epochs, verbose)

    def accuracy(self, X, y):
        return self.clf.get_accuracy(X, y)

    def get_parameters(self):
        return self.clf.get_parameters()
    
    def get_loss(self):
        return self.clf.loss


    def classify(self, X):
        '''
                Classify a set of pixels into red, green, or blue

                Inputs:
                  X: n x 3 matrix of RGB values
                Outputs:
                  y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE

        y = self.clf.classify(X)

        # YOUR CODE BEFORE THIS LINE
        ################################################################
        return y

class LogisticRegression():
    tolerance = 1e-5
    epoch = 100
    folder_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(folder_path, 'parameters/pixel_weights.npy')

    def __init__(self):
        '''
                Initilize your classifier with any parameters and attributes you need
        '''
        self.tolerance = LogisticRegression.tolerance
        self.epochs = LogisticRegression.epoch
        self.weights = np.load(LogisticRegression.model_path)
        self.color_classes = [1, 2, 3]
        self.class_labels = {1: 0, 2: 1, 3: 2}
        self.loss = []

    # Helper functions
    def calc_loss(self, y, y_pred):
        return -1 * np.mean(y*np.log(y_pred))

    def sigmoid(self, z):
        val = 1.0/(1.0 + np.exp(-z))
        return val

    def softmax(self, z):
        val = np.exp(z)/np.sum(np.exp(z), axis=1).reshape(-1, 1)
        return val

    def get_mini_batch(self, X, y, batchsize, shuffle=True):
        assert X.shape[0] == y.shape[0]
        indices = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, X.shape[0], batchsize):
            excerpt = indices[i:i + batchsize]
            yield X[excerpt], y[excerpt]

    def get_gradient(self, error, X):
        dw = np.dot(error.T, X)
        return dw

    def train(self, X, y, lr=0.001, batch_size=32, epochs=100, verbose=False):
        self.epochs = epochs
        self.color_classes = np.unique(y)
        # print(self.color_classes)
        # Create a dictionary for easier access (saves us time with adding +1 in the end)
        #self.class_labels = {1: 0, 2: 1, 3: 2}
        for i in range(len(self.color_classes)):
            self.class_labels[i+1] = i

        X = np.insert(X, 0, 1, axis=1)
        y = np.eye(len(self.color_classes))[np.vectorize(
            lambda c: self.class_labels[c])(y).reshape(-1)]
        # print(y)
        self.loss = []
        self.weights = np.zeros((len(self.color_classes), X.shape[1]))

        for i in range(self.epochs):
            # Save the initial loss for verification purposes
            # The very first value does not provide much information
            z = np.dot(X, self.weights.T).reshape(-1, len(self.color_classes))
            y_pred = self.softmax(z)
            self.loss.append(self.calc_loss(y, y_pred))
            # Create mini batches for better training
            for X_batch, y_batch in self.get_mini_batch(X, y, batch_size, shuffle=True):
                z = np.dot(X_batch, self.weights.T).reshape(-1,
                                                            len(self.color_classes))
                y_batch_pred = self.softmax(z)
                # Calculate the error for the pixels in the batch
                batch_error = y_batch - y_batch_pred

                # Run gradient descent algorithm
                dw = self.get_gradient(batch_error, X_batch)
                self.weights = self.weights + lr*dw

            # Save the model weights
            if np.abs(dw).max() < self.tolerance:
                np.save('parameters/pixel_weights.npy', self.weights)
                break
            if verbose:
                print("Error at epoch {} is {}.".format(i+1, batch_error))

            # Save the model weights
            np.save('parameters/pixel_weights.npy', self.weights)

    def get_accuracy(self, X, y):
        accuracy = np.mean(self.classify(X) == y)
        return accuracy

    def classify(self, X):
        '''
                Classify a set of pixels into red, green, or blue

                Inputs:
                  X: n x 3 matrix of RGB values
                Outputs:
                  y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE

        X = np.insert(X, 0, 1, axis=1)

        # Flatten it out to {nx3} dimensions where,
        # n - Number of pixels in the image and 3 are the number of classes
        z = np.dot(X, self.weights.T).reshape(-1, len(self.color_classes))
        self.probabilities = self.softmax(z)
        # Pick the class with highest probability
        y = np.vectorize(lambda c: self.color_classes[c])(
            np.argmax(self.probabilities, axis=1))

        # YOUR CODE BEFORE THIS LINE
        ################################################################
        return y

class GaussianNaiveBayes():
    tolerance = 1e-5
    epoch = 100
    folder_path = os.path.dirname(os.path.abspath(__file__))
    mean = os.path.join(folder_path, 'parameters/pixel_mean.npy')
    sigma = os.path.join(folder_path, 'parameters/pixel_var.npy')
    theta = os.path.join(folder_path, 'parameters/pixel_theta.npy')

    def __init__(self, num_classes):
        '''
                Initilize your classifier with any parameters and attributes you need
        '''
        self.pixel_mean = np.load(GaussianNaiveBayes.mean)
        self.pixel_var = np.load(GaussianNaiveBayes.sigma)
        self.prior = np.load(GaussianNaiveBayes.theta)
        self.num_classes = num_classes

    def get_parameters(self):
        return self.pixel_mean, self.pixel_var, self.prior
    
    def get_accuracy(self, X, y):
        accuracy = np.mean(self.classify(X) == y)
        return accuracy

    def train(self, X, y, lr=0.001, batch_size=32, epochs=100, verbose=False):

        print("Learning Gaussian parameters.....")
        if y.shape[0] != X.shape[0]:
            print("Size mismatch!")

        Xy = np.append(X, y.reshape(-1, 1), axis=1)
        X_list = []
        for i in range(self.num_classes):
          X_list.append(X[Xy[:, 3] == i+1])

        color_list = []
        for i in range(len(X_list)):
          color_list.append(X_list[i].shape[0])

        total_pixels = X.shape[0]
        x_temp = []
        for i in range(len(X_list)):
          x_temp.append(X_list[i])
        #x_temp = [X1, X2, X3]

        for i in range(self.num_classes):
            for j in range(X.shape[1]):
                self.pixel_mean[i][j] = np.mean(np.array(x_temp[i])[:, j])
                self.pixel_var[i][j] = np.var(np.array(x_temp[i])[:, j])

        prior_list = []
        for i in range(len(X_list)):
          prior_list.append(color_list[i]/total_pixels)

        #print(np.sum(prior_list))
        
        #prior_list = np.array(prior_list)
        np.save("parameters/pixel_mean.npy", self.pixel_mean)
        np.save("parameters/pixel_var.npy", self.pixel_var)
        np.save("parameters/pixel_theta.npy", prior_list)

    def classify(self, X):
        '''
                Classify a set of pixels into red, green, or blue

                Inputs:
                  X: n x 3 matrix of RGB values
                Outputs:
                  y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE
        # print(PixelClassifier.num_classes)
        # print(self.prior)
        # print(self.pixel_mean)
        # print(self.pixel_var)
        y = np.zeros((X.shape[0], 1))
        posterior_prob = np.zeros((self.num_classes, 1))
        post_log = np.zeros((self.num_classes, 1))
        # Using the learned parameters
        for row in range(X.shape[0]):
            for cls in range(self.num_classes):
                prob_class_color = 0
                #prob_class_colorLog = 0
                for col in range(X.shape[1]):
                    # exp_term = ((X[row,col] 
                    #                 - self.pixel_mean[cls,col])**2) / (2 * self.pixel_var[cls, col])
                    # norm_const = 1/ np.sqrt(2*np.pi*(self.pixel_var[cls, col]))
                    # prob_class_colorLog += norm_const * np.exp(-exp_term)
                    prob_class_color += np.log(self.pixel_var[cls,col]) + (((X[row,col] 
                                    - self.pixel_mean[cls,col])**2)/self.pixel_var[cls,col])
                    
                posterior_prob[cls] = prob_class_color + np.log(1/self.prior[cls]**2)
                # post_log[cls] = prob_class_colorLog * self.prior[cls]
                # print(post_log, posterior_prob)
                #print(posterior_prob[cls])
            y[row] = np.argmin(posterior_prob) + 1

        # YOUR CODE BEFORE THIS LINE
        ################################################################
        return y
