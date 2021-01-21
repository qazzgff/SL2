import numpy as np
from perceptron import Percetron

def load_data():
    data = np.loadtxt("zipcombo.dat")
    return data

# split data into train and test sets
def split(data, size):
    offset = int(len(data)*size)
    np.random.shuffle(data)
    test_set = data[:offset]
    train_set = data[offset:]
    return train_set, test_set

# split input vectors and labels
def get_label(full_data):
    y = full_data[:,0]
    x = np.delete(full_data,0,1)

    return x,y

data = load_data()
train_data,test_data = split(data, 0.2)
train_x,train_y = get_label(train_data)
test_x,test_y = get_label(test_data)


for i in range(len(train_y)):
    if train_y[i] != 1:
        train_y[i] = -1

p = Percetron(train_x, train_y, d=2)
for i in range(0,20):
    error = p.train(train_x, train_y)
    print(i)
    print(error)
