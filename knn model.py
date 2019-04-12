# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Import of numpy array and matplotlib to plot the graph
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import warnings
warnings.filterwarnings('ignore')

#importing file 
data_train = pd.read_table("train-data.txt",delim_whitespace=True,header=None)
data_test = pd.read_table("test-data.txt",delim_whitespace=True,header=None)

# #checking for any missung values in the data frame
# print((sum(data_train.isnull().sum())))
# print((sum(data_test.isnull().sum())))

#Created a separate dataframe for labels of training data 
train_data_label=data_train[1]


#Converted the train labels from dataframe to array
train_data_label=train_data_label.values

#Dropping the columns of labels from train data
data_train.drop(0,axis=1,inplace=True)
data_train.drop(1,axis=1,inplace=True)


#crated a separate dataframe for labels in test data
test_data_label=data_test[1]


#converted the test labels from dataframe to array
test_data_label=test_data_label.values

#Dropping the columns of lables from test data
data_test.drop(0,axis=1,inplace=True)
data_test.drop(1,axis=1,inplace=True)


#converting train and test data without labels to array
data_train=data_train.values
data_test=data_test.values



# Function for Euclidean distance calculation
#Note: Reference were taken from stackoverflow

def distance(x,y):   
    return np.sqrt(np.sum((x-y)**2))

        
# function to find nearest neighbours
import operator
def knn(test_data,train_data,k):
    dist=[]                        #dist- is a list which conatins neighbours and its corresponding distances from the test data
    length=len(train_data)
    for j in range(length):
        d=distance(train_data[j],test_data)
        dist.append((j,d))
    dist.sort(key=operator.itemgetter(1))
    neighbours=[]
    for x in range(k):
        neighbours.append(dist[x][0])
    return neighbours

    
#function to find the mode of the labels of the nearest neighbours
import statistics
def find_max_mode(list_data):
    list_table = statistics._counts(list_data)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = statistics.mode(list_data)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        max_mode = max(new_list) 
    return max_mode

#Function to find predictions of test data by K-NN model
#k - Number of nearest neighbours 
#a - Input Train_data
#b - Test data
#train_label 
def prediction(a,b,train_label, k):
    predictions = []
    for i in range(len(b)):
        neigh=knn(b[i],a,k)
        n_labels = []                                    #n_labels is a list which contains labels of the predicted neighbours
        n_labels = train_label[neigh]
        predictions.append(find_max_mode(n_labels))      #voting on the correct orientation
    return predictions



#Function to find Accuracy 
#Accuracy was found by comparing the predicted value by model and actual value on the test data
def accuracy(predicted,actual):
        count1=0
        for i in range(len(predicted)):
            if predicted[i]==actual[i]:
                count1+=1
            accuracy1=count1/len(predicted)
        return(accuracy1)
        
starttime=time.time() 
predicted=prediction(data_train,data_test, train_data_label, 41)
totaltime=time.time()-starttime
acc=(accuracy(predicted,test_data_label))*100
print("Value of accuracy for KNN-Model :%5f"% acc)
print("Time elapsed for KNN_Model :%5f"% totaltime)




#Code to observe k_value vs accuracy
#import time
#k_value=[]  
#time2=[]      
#for k in range(5,42,4):
    #starttime=time.time()
    #predicted=prediction(data_train,data_test, train_data_label, k) 
    #totaltime=time.time()-starttime
    #time2.append(totaltime)
    #k_value.append(k)
#print(k_value)
#print(time2)

#import matplotlib.pyplot as plt
#plt.ylabel('Time')
#plt.xlabel('K-nearest neighbors')
#plt.plot(k_value,time2)
#plt.title("K-value vs Time")
#plt.show()



#Code to view performance on varying data size

# from mpl_toolkits.mplot3d import Axes3D
# import time
# #calculation of performance for different data set
# pre=[]
# acc1=[]
# time1=[]
# data=[]
# for i in range(2311,(len(data_train))+1,4622):
#     starttime=time.time()        
#     predicted=prediction(data_train[:i,:],data_test, train_data_label, 41) 
#     pre.append(predicted)
#     totaltime=time.time()-starttime
#     time1.append(totaltime)
#     acc=accuracy(predicted,test_data_label)
#     acc1.append(acc)
#     data.append(i)
# print(data)
# print(acc1)
# print(time1)

# #plot of performance of model 

# fig=plt.figure()
# ax=fig.add_subplot(111,projection="3d")

# Axes3D.plot3D(ax,xs=data,ys=acc1,zs=time1,color="b")
# plt.title("data sets vs test accuracy vs time elapsed")
# plt.show()
 
    
 #Code to find classified and misclassified image
   
#print(test_data_label[1:10])
# labels=pd.read_csv("test-data.txt",delim_whitespace=True,header=None).iloc[:,0]
# k_val = []
# acc = []
# pred1=[]
# Misclassified=[]
# classified=[]
# p = prediction(data_train,data_test, train_data_label,41)
# for i in range(20):
#     if p[i]==test_data_label[i]:
#         classified.append(labels[i])
#     else:
#         Misclassified.append(labels[i])

# print(classified)
# print(Misclassified)