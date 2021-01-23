import numpy as np
from perceptron import Percetron
from numba import jit

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




def one_vs_all(train_x,train_y,d,test_x,test_y):
    # calss_num = 10
    epoch = 5
    # errors = np.zeros(epoch)
    Percetron_list = []
    
    # training
    for classes in range(0,10):
        # data preprocessing
        y_cur = train_y * 1
        for i in range(len(y_cur)):
            if y_cur[i] != classes:
                y_cur[i] = -1
            else:
                y_cur[i] = 1
        
        percetron = Percetron(train_x,y_cur,d)
        for ep in range(1,epoch+1):
            print('class: '+ str(classes) + ' epoch: '+ str(ep)+' d= '+ str(d) )
            percetron.train(train_x,y_cur)
            
        
        Percetron_list.append(percetron)
    
    # testing
    # training errors
    train_error = 0
    for i in range(len(train_y)):
        confidence = np.zeros(10)
        confidence = confidence.tolist()
        for j in range(0,10):
            confidence[j] = Percetron_list[j].predict(train_x[i])
        predict_label = confidence.index(max(confidence))
        if predict_label != train_y[i]:
            train_error = train_error + 1
    
    # test error
    test_error = 0
    for i in range(len(test_y)):
        confidence = np.zeros(10)
        confidence = confidence.tolist()
        for j in range(0,10):
            confidence[j] = Percetron_list[j].predict(test_x[i])
        predict_label = confidence.index(max(confidence))
        if predict_label != test_y[i]:
            test_error = test_error + 1
    
    print('d= '+str(d)+' train error: '+str(train_error)+' test error: '+str(test_error))


    return train_error,test_error





trainerror_mean = np.zeros(7)
trainerror_std = np.zeros(7)        
testerror_mean = np.zeros(7)
testerror_std = np.zeros(7)
num_runs = 20  

        
for d in range(1,8):    
    train_errors = np.zeros(num_runs)
    test_errors = np.zeros(num_runs)

    for run in range(0,num_runs):
        print('run: '+ str(run))
        data = load_data()
        train_data,test_data = split(data, 0.2)
        train_x,train_y = get_label(train_data)
        test_x,test_y = get_label(test_data)

        train_errors[run],test_errors[run] = one_vs_all(train_x,train_y,d,test_x,test_y)
    
    trainerror_mean[d-1] = train_errors.mean()
    trainerror_std[d-1] = train_errors.std()
    testerror_mean[d-1] = test_errors.mean()
    testerror_std[d-1] = test_errors.std()

for i in range(len(trainerror_mean)):
    print('d='+str(i+1)+' mean train error: '+str(trainerror_mean[i])+' ± '+str(trainerror_std[i]))
    print('d='+str(i+1)+' mean test error: '+str(testerror_mean[i])+' ± '+str(testerror_std[i]))






