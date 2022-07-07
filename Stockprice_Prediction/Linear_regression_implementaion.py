#!/usr/bin/env python
# coding: utf-8

# In[14]:


#!/usr/bin/env python
# coding: utf-8

# import packages
# Note: You cannot import any other packages!
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import random


# Global attributes
# Do not change anything here except TODO 1 
StudentID = '106062304' # TODO 1 : Fill your student ID here
input_dataroot = 'input.csv' # Please name your input csv file as 'input.csv'
output_dataroot =StudentID+'_basic_prediction.csv'
# Output file will be named as '[StudentID]_basic_prediction.csv'

input_datalist =  [] # Initial datalist, saved as numpy array
output_datalist =  [] # Your prediction, should be 20 * 2 matrix and saved as numpy array
                      # The format of each row should be [Date, TSMC_Price_Prediction] 
                      # e.g. ['2021/10/15', 512]

# You can add your own global attributes here
# Read input csv to datalist
with open(input_dataroot, newline='') as csvfile:
    input_datalist = np.array(list(csv.reader(csvfile)))
#     print(input_datalist)
#     print(input_datalist.shape)
 


# In[15]:


def SplitData():
# TODO 2: Split data, 2021/10/15 ~ 2021/11/11 for testing data, and the other for training data and validation data 
    data = []
    train = input_datalist[129:189]#前60天等下都下去訓練
    test = input_datalist[189:]
    validate = input_datalist[169:189]
    data.append(train)
    data.append(test)
    data.append(validate)
    return data


# In[16]:


def PreprocessData():
# TODO 3: Preprocess your data  e.g. split datalist to x_datalist and y_datalist
    x_datalist=[]
    y_datalist=[]
    vx_datalist=[]
    vy_datalist=[]
    data=[]
    
    for i in range(0,60):
        x_datalist.append(int(SplitData()[0][i][1]))#put in 60 training data
        y_datalist.append(int(SplitData()[0][i][2]))
        
    for i in range(0,20):   
        vx_datalist.append(int(SplitData()[2][i][1]))#put in 20 validate data
        vy_datalist.append(int(SplitData()[2][i][2]))

    data.append(x_datalist)
    data.append(y_datalist)
    data.append(vx_datalist)
    data.append(vy_datalist)
    return(data)


# In[17]:


def Regression():
# TODO 4: Implement regression
    data = []
    x=np.array(PreprocessData()[0])
    y=np.array(PreprocessData()[1])#old y of training data

    newx=np.concatenate((np.ones((x.shape[0],1)),x[:,np.newaxis]),axis=1)
    newy=y[:,np.newaxis]
    beta=np.matmul(np.matmul(np.linalg.inv(np.matmul(newx.T,newx)),newx.T),newy)
    
    valx = PreprocessData()[2]
    valy = PreprocessData()[3]#validate data for test
     
    xs=valx
    ys=beta[0]+beta[1]*xs #ys = new y of training data
    
    plt.scatter(valx,valy,s=100)
    plt.plot(xs,ys,'r',linewidth=3)
    plt.xlabel('x軸',fontsize=20)
    plt.ylabel('y軸',fontsize=20,rotation=0)
#     plt.savefig('ols_datapoint')

    data.append(beta)
    data.append(valy)#old 
    data.append(ys)#new

    
    return data


# In[18]:


def CountLoss(new,old):
# TODO 5: Count loss of training and validation data 
    loss = 1/len(old) * np.sum((new-old)**2)
    return loss


# In[19]:


def MakePrediction(test):
# TODO 6: Make prediction of testing data 
   
    a = Regression()[0][1]#beta weight
    b = Regression()[0][0]#beta bias
    y = a*test+b

    return y


# In[20]:


# TODO 7: Call functions of TODO 2 to TODO 6, train the model and make prediction
print("OLS_countloss:")
print(CountLoss(Regression()[2],Regression()[1]))# 60:63.9 90:82  30:152


# In[21]:


def gradient():
    data=[]
    rate=0.0000000001

    x=np.array(PreprocessData()[0])
    y=np.array(PreprocessData()[1])
    
    theta = np.random.randn(2,1)
    #加入w0
    newx=np.concatenate((np.ones((x.shape[0],1)),x[:,np.newaxis]),axis=1)
    newy=y[:,np.newaxis]
      
    for i in range(1000):#1000次
        predict_y = np.dot(newx,theta)
        err = newy-predict_y
        gradient = 2*np.dot(newx.T,err)
        theta += rate*gradient
        #newx.shape = (186,2) 
        #newy.shape = (186,1)
        #theta.shape = (2,1)
        #err.shape = (186,1) 
        #gradient.shape = (2,1)
    
    valx = PreprocessData()[2]
    valy = PreprocessData()[3]
        
    xs=valx
    ys=theta[0]+theta[1]*xs
    
    plt.scatter(valx,valy,s=100)
    plt.plot(xs,ys,'r',linewidth=3)
    plt.xlabel('x軸',fontsize=20)
    plt.ylabel('y軸',fontsize=20,rotation=0)
#     plt.savefig('grad_datapoint')

    data.append(theta)
    data.append(valy)
    data.append(ys)
    
    return data


# In[22]:


print("Gradient_descent_countloss:")
print(CountLoss(gradient()[2],gradient()[1]))#  90:77.628 60:80.28  30:190


# In[23]:


test=[]
data = SplitData()[1]
for price in data:
     test.append(int(price[1]))


# In[24]:


MakePrediction(test)


# In[25]:


# Write prediction to output csv
with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
#     print(input_datalist[186:,1])
    
    tsmc = list(MakePrediction(test))
    
    output_datalist = input_datalist
    output_datalist = np.delete(output_datalist,2,axis=1)
    
    for i in range(0,20):
        output_datalist[i,0]=output_datalist[i+189,0]
        output_datalist[i,1]=str(tsmc[i])
        
    for i in range(20,209):
        output_datalist = np.delete(output_datalist,20,axis=0)  
        
    print(output_datalist)
    for row in output_datalist:
        writer.writerow(row)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




