'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
import os


class PixelClassifier():
    tolerance = 1e-5
    epoch = 100
    folder_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(folder_path, 'parameters/pixel_weights.npy')

    def __init__(self):
        '''
                Initilize your classifier with any parameters and attributes you need
        '''
        self.tolerance = PixelClassifier.tolerance
        self.epochs = PixelClassifier.epoch
        self.weights = np.load(PixelClassifier.model_path)
        self.color_classes = [1, 2, 3]

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
        self.class_labels = {1: 0, 2: 1, 3: 2}
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

    def score(self, X, y):
        score = np.mean(self.classify(X) == y)
        return score

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


class BayesianCalssifier():

    num_classes = 3  # One each for Red, Green and Blue channels

    def __init__(self):
        '''
                Initilize your classifier with any parameters and attributes you need
        '''
        self.pixel_mean = np.load("parameters/pixel_mean.npy")
        self.pixel_var = np.load("parameters/pixel_var.npy")
        self.prior = np.load("parameters/priors.npy")

    def set_distibution_parameters(self, X, y, X1, X2, X3, y1, y2, y3, verbose=False):

        print("Learning Gaussian parameters.....")
        if y.shape[0] != X.shape[0]:
            print("Size mismatch!")

        red_pixels = X1.shape[0]
        green_pixels = X2.shape[0]
        blue_pixels = X3.shape[0]
        total_pixels = X.shape[0]

        x_temp = [X1, X2, X3]

        for i in range(PixelClassifier.num_classes):
            for j in range(X.shape[1]):
                self.pixel_mean[i][j] = np.mean(np.array(x_temp[i])[:, j])
                self.pixel_var[i][j] = np.var(np.array(x_temp[i])[:, j])

        self.red_prior = red_pixels/total_pixels
        self.green_prior = green_pixels/total_pixels
        self.blue_prior = blue_pixels/total_pixels

        self.prior = np.array(
            [self.red_prior, self.green_prior, self.blue_prior])
        np.save("parameters/pixel_mean.npy", self.pixel_mean)
        np.save("parameters/pixel_var.npy", self.pixel_var)
        np.save("parameters/priors.npy", self.prior)
        #print(self.pixel_mean, self.pixel_var, self.prior)
        print("Learning Gaussian parameters done....")

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
        posterior_prob = np.zeros((PixelClassifier.num_classes, 1))
        # Using the learned parameters
        for row in range(X.shape[0]):
            for cls in range(PixelClassifier.num_classes):
                prob_class_color = 0
                for col in range(X.shape[1]):
                    exp_term = ((X[row, col] - self.pixel_mean[cls, col])
                                ** 2 / (self.pixel_var[cls, col]))
                    norm_const = (0.5*np.pi*(self.pixel_var[cls, col]))
                    prob_class_color += norm_const * np.exp(-exp_term)
                posterior_prob[cls] = prob_class_color / self.prior[cls]
                # print(posterior_prob[cls])
            y[row] = np.argmax(posterior_prob) + 1
        # YOUR CODE BEFORE THIS LINE
        ################################################################
        return y
