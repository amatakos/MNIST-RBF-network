import sys
import numpy as np
from sklearn.cluster import KMeans
import math

class RBFN(object):

    def __init__(self, hidden_shape):
        """ radial basis function network
        # Arguments
            input_shape: dimension of the input data
            e.g. scalar functions have should have input_dimension = 1
            hidden_shape: the number
            hidden_shape: number of hidden radial basis functions,
            also, number of centers.
        """
        self.hidden_shape = hidden_shape
        self.sigma = None
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        return np.exp(-self.sigma*np.linalg.norm(center-data_point)**2)

    def _logistic_function(self, center, data_point):
        return 1/(1 + np.exp(-self.sigma*np.linalg.norm(center-data_point)))

    def _quadratic_function(self, center, data_point):
        return np.linalg.norm(center-data_point)**2

    def _calculate_matrix(self, X):
        """ Calculates matrix using a kernel_function
        # Arguments
            X: Training data
        # Input shape
            (num_data_samples, input_shape)
        # Returns
            G: matrix
        """
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                #G[data_point_arg, center_arg] = self._kernel_function(
                #        center, data_point)
                G[data_point_arg, center_arg] = self._logistic_function(
                        center, data_point)
                #G[data_point_arg, center_arg] = self._quadratic_function(
                #        center, data_point)
        return G

    def _select_centers(self, X):
        random_args = np.random.choice(len(X), self.hidden_shape, replace = False)
        centers = X[random_args]
        print('Randomly selecting centers from dataset')
        return centers

    def _K_means(self, X):
        km = KMeans(n_clusters = self.hidden_shape, max_iter = 100)
        km.fit(X)
        cent = km.cluster_centers_
        print('Finding centers using K-means algorithm')
        return cent

    def _determine_sigma(self, X, cent):
        max = 0
        for i in range(self.hidden_shape):
        	for j in range(self.hidden_shape):
        		d = np.linalg.norm(cent[i]-cent[j])
        		if (d > max):
        			max = d
        d = max
        sigma = d/math.sqrt(2*self.hidden_shape)
        return sigma

    def fit(self, X, Y):
        """ Fits weights using linear regression
        # Arguments
            X: training samples
            Y: targets
        # Input shape
            X: (num_data_samples, input_shape)
            Y: (num_data_samples, input_shape)
        """
        #self.centers = self._select_centers(X) #random
        self.centers = self._K_means(X)
        self.sigma = self._determine_sigma(X, self.centers)
        #print("Radial function: Gaussian")
        print("Radial function: Logistic")
        #print("Radial function: Quadratic")
        print("sigma = " + str(self.sigma))
        G = self._calculate_matrix(X)
        D = np.zeros((Y.shape[0], 10))
        for i,v in enumerate(Y):
            D[i,v] = 1
        self.weights = np.dot(np.linalg.pinv(G), D)

    def predict(self, X):
        """
        # Arguments
            X: test data
        # Input shape
            (num_test_samples, input_shape)
        """
        G = self._calculate_matrix(X)
        predictions = np.dot(G, self.weights)
        predictions = np.argmax(predictions,axis=1)
        return predictions
