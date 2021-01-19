import numpy as np

def load_data():
    data = np.loadtxt("zipcombo.dat")
    y = data[:,0]
    x = np.delete(data,0,1)
    
    return x,y

def poly_kernel(x, y, d):
    K = np.dot(x, y.T)
    K = np.power(K, d)
    return K



x_trian,y_trian = load_data()

