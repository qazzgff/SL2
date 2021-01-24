import numpy as np
from perceptron import Percetron
from numba import jit
import time
import math

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

def merge_list(lists,i):
    t = []
    for j in range(5):
        if j != i:
            if len(t) != 0:
                add = np.array(lists[j])
                t= np.vstack((t,add))
            else:
                t = lists[j]
                t = np.array(t)
    
    return t



def one_vs_all_m(train_x,train_y,d,test_x,test_y,k):
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
        
        # percetron = Percetron(train_x,y_cur,d)
        percetron = Percetron(train_x,y_cur,d,kernel=k)
        # for ep in range(1,epoch+1):
        #     print('class: '+ str(classes) + ' epoch: '+ str(ep)+' d= '+ str(d) )
            # percetron.train(train_x,y_cur)
        print('calss: '+str(classes))
        percetron.train(train_x,y_cur,epoch)
        
            
        
        Percetron_list.append(percetron)


    train_error = 0
    confidence = np.zeros(10)
    confidence = confidence.tolist()
    for j in range(0,10):
        confidence[j] = Percetron_list[j].predict(train_x)

    confidence = np.array(confidence).T
    confidence = confidence.tolist()
    
    for i in range(len(train_y)):
        c = confidence[i]
        predict_label = c.index(max(c))
        if predict_label != train_y[i]:
            train_error = train_error + 1
    
    # test error
    test_error = 0
    confidence = np.zeros(10)
    confidence = confidence.tolist()
    for j in range(0,10):
        confidence[j] = Percetron_list[j].predict(test_x)
    
    confidence = np.array(confidence).T
    confidence = confidence.tolist()
    for i in range(len(test_y)):
        c = confidence[i]
        predict_label = c.index(max(c))
        if predict_label != test_y[i]:
            test_error = test_error + 1
    

    train_error = train_error/(len(train_y))
    test_error = test_error/(len(test_y))
    print('d= '+str(d)+' train error: '+str(train_error)+' test error: '+str(test_error))
    return train_error,test_error

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
        
        # percetron = Percetron(train_x,y_cur,d)
        percetron = Percetron(train_x,y_cur,d,kernel='gaul')
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

    train_error = train_error/(len(train_y))
    test_error = test_error/(len(test_y))
    return train_error,test_error

def one_vs_all_2(train_x,train_y,d,test_x,test_y):
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
    
    print('d= '+str(d)+' test error: '+str(test_error))


    return test_error

def one_vs_all_confusion(train_x,train_y,d,test_x,test_y):
    # calss_num = 10
    epoch = 5
    # errors = np.zeros(epoch)
    Percetron_list = []

    confusion_mat = np.zeros(shape=(10,10))
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
    
    
    # produce confusion matrix
    for i in range(len(test_y)):
        confidence = np.zeros(10)
        confidence = confidence.tolist()
        for j in range(0,10):
            confidence[j] = Percetron_list[j].predict(test_x[i])
        predict_label = confidence.index(max(confidence))

        if predict_label != test_y[i]:
            confusion_mat[int(test_y[i])][int(predict_label)] += 1 
    

    return confusion_mat

def q1_1():
    trainerror_mean = np.zeros(7)
    trainerror_std = np.zeros(7)        
    testerror_mean = np.zeros(7)
    testerror_std = np.zeros(7)
    num_runs = 20  
    data = load_data()
            
    for d in range(1,8):    
        train_errors = np.zeros(num_runs)
        test_errors = np.zeros(num_runs)

        for run in range(0,num_runs):
            print('run: '+ str(run))
            # data = load_data()
            train_data,test_data = split(data, 0.2)
            train_x,train_y = get_label(train_data)
            test_x,test_y = get_label(test_data)

            train_errors[run],test_errors[run] = one_vs_all_m(train_x,train_y,d,test_x,test_y,k='poly')
        
        trainerror_mean[d-1] = train_errors.mean()
        trainerror_std[d-1] = train_errors.std()
        testerror_mean[d-1] = test_errors.mean()
        testerror_std[d-1] = test_errors.std()

    for i in range(len(trainerror_mean)):
        print('d='+str(i+1)+' mean train error: '+str(trainerror_mean[i])+' ± '+str(trainerror_std[i]))
        print('d='+str(i+1)+' mean test error: '+str(testerror_mean[i])+' ± '+str(testerror_std[i]))


def q1_2():
    num_runs = 20
    data = load_data()
    best_d_list = np.zeros(num_runs)
    test_error = np.zeros(num_runs)
    
    
    for run in range(0,num_runs):
        train_data,test_data = split(data, 0.2)
        test_x,test_y = get_label(test_data)
        test_errors_list = np.zeros(7)
        for d in range(1,8):
            print('run: '+ str(run)+' d= '+str(d))
            # split training data into 5 folds
            traindata_list = []
            length = len(train_data)
            n = 5
            for i in range(n):
                one_list = train_data[math.floor(i / n * length):math.floor((i + 1) / n * length)]
                traindata_list.append(one_list)
            
            test_errors = np.zeros(5)
            for fold in range(0,5):
                print('fold: '+ str(fold))
                cv_test_x,cv_test_y = get_label(traindata_list[fold])
                cv_train = merge_list(traindata_list,fold)
                cv_train_x,cv_train_y = get_label(cv_train)

                test_errors[fold] = one_vs_all_2(cv_train_x,cv_train_y,d,cv_test_x,cv_test_y)
            
            test_errors_list[d-1] = test_errors.mean()
        
        # best d
        test_errors_list = test_errors_list.tolist()
        best_d = test_errors_list.index(min(test_errors_list)) + 1
        best_d_list[run] = best_d

        # test error
        trainx,trainy = get_label(train_data)
        testserror = one_vs_all_2(trainx,trainy,best_d,test_x,test_y)
        test_error[run] = testserror
    
    print('test error: '+str(test_error.mean())+' ± '+str(test_error.std()))
    print('best d: '+ str(best_d_list.mean())+' ± '+ str(best_d_list.std()))



def q1_3():
    num_runs = 20
    data = load_data()
    best_d_list = np.zeros(num_runs)
    confusion_matrix = np.zeros(shape=(10,10))
    confusion_matrix_std = np.zeros(shape=(10,10))
    confusion_matrix_list = np.zeros(shape=(num_runs,10,10))
    
    for run in range(0,num_runs):
        print('run: '+str(run))
        train_data,test_data = split(data, 0.2)
        test_x,test_y = get_label(test_data)
        test_errors_list = np.zeros(7)
        
        for d in range(1,8):
            print('run: '+ str(run)+' d= '+str(d))
            # split training data into 5 folds
            traindata_list = []
            length = len(train_data)
            n = 5
            for i in range(n):
                one_list = train_data[math.floor(i / n * length):math.floor((i + 1) / n * length)]
                traindata_list.append(one_list)
            
            test_errors = np.zeros(5)
            for fold in range(0,5):
                print('fold: '+ str(fold))
                cv_test_x,cv_test_y = get_label(traindata_list[fold])
                cv_train = merge_list(traindata_list,fold)
                cv_train_x,cv_train_y = get_label(cv_train)

                test_errors[fold] = one_vs_all_2(cv_train_x,cv_train_y,d,cv_test_x,cv_test_y)
            
            test_errors_list[d-1] = test_errors.mean()
        
        # best d
        test_errors_list = test_errors_list.tolist()
        best_d = test_errors_list.index(min(test_errors_list)) + 1
        best_d_list[run] = best_d
        
        # calculating the confusion error

        trainx,trainy = get_label(train_data)
        tmp = one_vs_all_confusion(trainx,trainy,best_d_list[run],test_x,test_y)
        confusion_matrix += tmp
        confusion_matrix_list[run] = tmp
    
    confusion_matrix = confusion_matrix*(1/num_runs)
    for i in range(0,10):
        for j in range(0,10):
            cur_list = np.zeros(num_runs)
            for r in range(num_runs):
                cur_list[r] = confusion_matrix_list[r][i][j]
            
        std = cur_list.std()
        confusion_matrix_std[i][j] = std
               


    print('confusion_matrix:')
    print(confusion_matrix)
    print('confusion_matrix_std:')
    print(confusion_matrix_std)        


        

def q1_5():
    trainerror_mean = np.zeros(7)
    trainerror_std = np.zeros(7)        
    testerror_mean = np.zeros(7)
    testerror_std = np.zeros(7)
    num_runs = 20  
    data = load_data()
    
    
    S = np.array([0.001,0.003,0.005,0.008,0.01,0.02,0.03])
    for d in range(1,8):    
        train_errors = np.zeros(num_runs)
        test_errors = np.zeros(num_runs)
        c = S[d-1]

        
        for run in range(0,num_runs):
            print('run: '+ str(run))
            # data = load_data()
            train_data,test_data = split(data, 0.2)
            train_x,train_y = get_label(train_data)
            test_x,test_y = get_label(test_data)

            train_errors[run],test_errors[run] = one_vs_all_m(train_x,train_y,c,test_x,test_y,k='gaul')
        
        trainerror_mean[d-1] = train_errors.mean()
        trainerror_std[d-1] = train_errors.std()
        testerror_mean[d-1] = test_errors.mean()
        testerror_std[d-1] = test_errors.std()
    
    

    for i in range(len(trainerror_mean)):
        print('c='+str(S[i])+' mean train error: '+str(trainerror_mean[i])+' ± '+str(trainerror_std[i]))
    for i in range(len(trainerror_mean)):
        print('c='+str(S[i])+' mean test error: '+str(testerror_mean[i])+' ± '+str(testerror_std[i]))


def q1_5_2():


    









        
        
        





if __name__ == '__main__':
    # q1_1()
    # q1_2()
    # q1_3()
    q1_5()




