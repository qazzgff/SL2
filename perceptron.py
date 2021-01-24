import numpy as np
from numba import jit
import time
from sklearn.metrics.pairwise import euclidean_distances
import numexpr as ne

class Percetron(object):
    def __init__(self, train_data, true_labels, d, lr = 1, kernel = 'poly'):
        self.w   =   np.zeros(len(train_data))
        self.lr =   lr
        self.train_data   = train_data
        self.true_labels  = true_labels
        self.d = d
        self.kernel = kernel
        self.errors = 0

    
    def predict(self, input_data):
        if self.kernel == 'poly':
            K = np.power(np.dot(self.train_data, input_data.T), self.d)
            res = np.dot(self.w, K)
        if self.kernel == 'gaul':
            # input_data_matrix = np.zeros(shape=(len(self.train_data),len(input_data)))
            # for i in range(len(input_data_matrix)):
            #     input_data_matrix[i] = input_data
            # K = np.exp(-1*self.d*np.linalg.norm(self.train_data-input_data_matrix,axis=1)**2)
            K = self.gaussian_kernel(input_data, self.train_data,self.d)
            res = np.dot(self.w, K.T)
        return res

    # @jit(nopython=True)
    def train(self, input_data, true_labels,epochs):
        self.errors = 0
        arraysize = len(input_data)
        if self.kernel == 'poly':
            K = np.power(np.dot(self.train_data, self.train_data.T), self.d)
        if self.kernel == 'gaul':
                K = self.gaussian_kernel(self.train_data, self.train_data,self.d) 
                # dist = euclidean_distances(self.train_data, self.train_data)
                # K = np.exp(self.d * np.power(dist, 2))
        
        for ep in range(epochs):
            print('d= '+str(self.d)+' epoch: '+str(ep))
            for i in range(0,arraysize):
                k = K[i]
                cur_predict = np.dot(self.w, k.T)
                cur_predict = np.sign(cur_predict)
                if cur_predict != true_labels[i]:
                    self.errors = self.errors + 1
                    self.w[i] = self.w[i] + true_labels[i]
        
        
        return self.errors

    def gaussian_kernel(self, p, q, c):
            X = p
            X_norm = np.sum(X ** 2, axis = -1)
            Y = q
            Y_norm = np.sum(Y ** 2, axis = -1)
            K = ne.evaluate('exp(-g * (A + B - 2 * C))', {
                'A' : X_norm[:,None],
                'B' : Y_norm[None,:],
                'C' : np.dot(X, Y.T),
                'g' : c
            })
            return K