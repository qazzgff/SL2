import numpy as np

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
            K = np.power(np.dot(self.train_data, input_data), self.d)
            res = np.dot(self.w, K.T)
            res = np.sign(res)
        return res
    
    def train(self, input_data, true_labels):
        self.errors = 0
        arraysize = len(input_data)
        for i in range(0,arraysize):
            cur_predict = self.predict(input_data[i])
            if cur_predict != true_labels[i]:
                self.errors = self.errors + 1
                self.w[i] = self.w[i] + true_labels[i]
        
        return self.errors
