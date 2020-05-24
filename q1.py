###########################Initialization
###########################Question 1
#this code is the NN algorithm developed without using library; the output of this file is the count and correct number of matching;
#need to select the m (number of training image) first which is on line 13
from scipy.io import loadmat
import time
import collections
import numpy as np
from numpy.linalg import norm
import random
import pandas
import cv2
lst_m = [10, 20, 30, 40, 50]  # number of training image
m= lst_m[0]  # initialization of the nummber of training set

start = time.time()


import math
loadtemp = loadmat('YaleB_32x32.mat')
gnd = loadtemp["gnd"]  # 2414 X 1
fea = loadtemp["fea"]  # (2414, 1024)


fea_copy2 = np.copy(fea)


n_training = []
n_testing = []
unique, counts = np.unique(gnd, return_counts=True)
D = dict(zip(unique, counts))
for i in range(len(counts)):
    n_training.append(m)  # random.choice(m)
for i in range(len(counts)):
    diff_temp = int(D[i+1]) - int(n_training[i])
    n_testing.append(diff_temp)
# select index in training
########################################################
def list_gen(n):
    lst = []
    for i in range(int(n)):
        lst.append(i + 1)
    return lst
def rand_index():
    for Dic in D:
        card = list_gen(D[Dic])
        lst_training = []
        lst_testing = []
        for item in n_training:
            # temp1 = abs(int(Dic) - int(item))+1
            random.shuffle(card)
            lst_training.append(card[:int(item)])
            lst_testing.append(card[int(item):])
            break
        conn.append(lst_training)
        conn1.append(lst_testing)
    index_training = pandas.DataFrame(conn)   # index +1
    index_testing = pandas.DataFrame(conn1)  # index +1
    return conn, conn1, index_testing , index_training
#######training set
def segmen():
    x = []
    y = []
    m=0
    n=0
    for Dic in D:
        x.append(m)
        m += D[Dic]
        n = m
        y.append(n)
    return x, y

def training_set():
    def temp(x,y):
        lst_training = []
        lst_training0 = []
        for Dic in D:
            lst = fea[x: y]
            for i in range(38):
                for j in range(n_training[i]):  # length 10
                    try:
                        lst_training.append(lst[conn[i][0][j]])  # fga
                    except:
                        pass
                break
            lst_training0.append(lst_training)
            break
        return lst_training0

    def temptest(x,y):
        lst_testing = []
        lst_testing0 = []
        for Dic in D:
            lst = fea[x: y]
            for i in range(38):
                for j in range(n_testing[i]):  # length 10
                    try:
                        lst_testing.append(lst[conn1[i][0][j]])  # fga
                    except:
                        pass
                break
            lst_testing0.append(lst_testing)
            break
        return lst_testing0

    lst1 = []
    lst2 = []

    for i in range(38):
        x, y = segmen()
        lst_training = temp(x[i],y[i])
        lst_testing = temptest(x[i],y[i])
        lst1 += lst_training
        lst2 += lst_testing

    return lst1, lst2

def df_training(index_training, index_testing):
    pass
def index():
    pass

def l2_norm(x,y):
    lst = []
    n = 0
    for i in range(0,len(x)):
        lst.append(abs(int(x[i])-int(y[i])))
    n = norm(np.array(lst))
    return n
def k_means(x,training_set, kn = 1): # for one data
    lst = []
    ni = []
    voting =0
    for i in range(len(training_set)):
        for j in range(len(training_set[i])):
            k = l2_norm(training_set[i][j],x)
            lst.append([k,i,j])
    y = np.array(lst)
    for i in range(int(kn)):
        x = y[np.argmin(y, axis=0)[0], 1]
        ni.append(x)
        index = np.argmin(y, axis=0)[0]
        y = np.delete(y, index, axis=0)
    return ni

def k_means_testing(testing_set):
    count = 0
    correct = 0
    for i in range(len(testing_set)):
        for j in range(len(testing_set[i])):
            x = testing_set[i][j]
            if k_means(x,training_set)[0] == i:
                correct += 1
            count +=1
        break  ###
    return count, correct

global conn
conn = []
global conn1
conn1 = []
conn, conn1, index_testing, index_training = rand_index()
training_set, testing_set = training_set()
count, correct = k_means_testing(testing_set)
print(count)
print(correct)
end = time.time()
print("time",end - start)


# for i in range(38):
#     training.append(conn[i])
#fea[i]

# def main():
# df_training.to_excel("test.xlsx")
# df_testing = pandas.DataFrame(lst_testing)






