'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np

#This class is the same class as the one used in problem 1. 
#Added a duplicate file to avoid confusion with number of class
#Difference is only in the number of classes
class PixelClassifier():
  tolerance = 1e-5
  epoch = 100
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    self.tolerance = PixelClassifier.tolerance
    self.epochs = PixelClassifier.epoch
    self.color_classes = [1, 2, 3, 4, 5] # {1: Bin-blue, 2: Other blue, 3:Red/Brown, 4: Green, 5: Black/Gray}
    
  #Helper functions
  def calc_loss(self, y, y_pred):
    return -1 * np.mean(y*np.log(y_pred))
  
  def sigmoid(self, z):
    val = 1.0/(1.0 + np.exp(-z))
    return val

  def softmax(self, z):
    val = np.exp(z)/np.sum(np.exp(z), axis=1).reshape(-1,1)
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
  
  #Helper function to check the accuracy
  def get_accuracy(self, X, y):
    accuracy = np.mean(self.classify(X) == y)
    return accuracy
  
  def get_weights(self):
    return self.weights

  def train(self, X, y, lr=0.001, batch_size=32, verbose=False):
    self.color_classes = np.unique(y)
    print(self.color_classes)
    #Create a dictionary for easier access (saves us time with adding +1 in the end)
    self.class_labels = {1:0, 2:1, 3:2, 4:3, 5: 4}
    X = np.insert(X, 0, 1, axis=1)
    y = np.eye(len(self.color_classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]
    #print(y)
    self.loss = []
    self.weights = np.zeros((len(self.color_classes), X.shape[1]))
   
   #Loop through the epochs
    for i in range(self.epochs):
      #Save the initial loss for verification purposes
      #The very first value does not provide much information
      z = np.dot(X, self.weights.T).reshape(-1,len(self.color_classes))
      y_pred = self.softmax(z)
      self.loss.append(self.calc_loss(y, y_pred))
      #Create mini batches for better training
      for X_batch, y_batch in self.get_mini_batch(X, y, batch_size, shuffle=True):
        z = np.dot(X_batch, self.weights.T).reshape(-1, len(self.color_classes))
        y_batch_pred = self.softmax(z)
        #Calculate the error for the pixels in the batch
        batch_error = y_batch - y_batch_pred

        #Run gradient descent algorithm
        dw = self.get_gradient(batch_error, X_batch)
        self.weights = self.weights + lr*dw

      #Save the model weights
      if np.abs(dw).max() < self.tolerance:
        np.save('parameters/classifier_weights.npy', self.weights)
        break
      if verbose:
        print("Epoch {} completed".format(i+1))

    #Save the model weights
    np.save('parameters/classifier_weights.npy', self.weights)

  #Unused function, copied from problem 1 solution, keeping it for consistency
  def classify(self,X):
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

    #Flatten it out to {nx3} dimensions where,
    #n - Number of pixels in the image and 3 are the number of classes
    z = np.dot(X, self.weights.T).reshape(-1,len(self.color_classes))
    self.probabilities = self.softmax(z)
    #Pick the class with highest probability
    y = np.vectorize(lambda c: self.color_classes[c])(np.argmax(self.probabilities, axis=1))

    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y


