#!/usr/bin/env python
# coding: utf-8

# In[455]:


from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import csv

path= 'input.csv' 
outpath='106062304_bonus_prediction.csv'
input_datalist = []
output_datalist = []
with open(path, newline='') as csvfile:
    input_datalist = np.array(list(csv.reader(csvfile)))
    
x=[]
for i in range(0,209):
     x.append(int(input_datalist[i][2]))
dataset=np.array(x)
dataset = dataset[:,np.newaxis]


# In[456]:


scaler = MinMaxScaler(feature_range=(0,1)).fit(dataset)
process = scaler.transform(dataset)


# In[457]:


data = int(len(dataset)) - 20
train_data = process[0:int(data), :]

x_train = []
y_train = []

for i in range(80, len(train_data)):#60~189
    x_train.append(train_data[i-80:i-20,0])#拿60個數據（0~60,1~61) 資料做位移展開
    y_train.append(train_data[i,0])#未來一天的數據
        
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))



# In[462]:


from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout

#LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[463]:


valx = process[0:int(data), :]
x_t = []
validation = dataset[60:data, :]


for i in range(60, len(valx)):
    x_t.append(valx[i-60:i, 0])
    
x_t = np.array(x_t)
x_t = np.reshape(x_t, (x_t.shape[0], x_t.shape[1], 1 ))

out = model.predict(x_t)
out = scaler.inverse_transform(out)

rmse = np.sqrt(np.mean(((out - validation) ** 2)))
mape = np.mean(np.abs((validation - out)/validation))*100
print(rmse)
print(mape)


# In[464]:


valx = process[data - 60:, :]
# print(test_data)
y_test = dataset[data:, :]
predictions=[]


for i in range(60, len(valx)):
    x_t = []
    x_t.append(valx[i-60:i, 0])
    #testdate分批(前60天數據)
    
    x_t = np.array(x_t)
    x_t = np.reshape(x_t, (x_t.shape[0], x_t.shape[1], 1 ))

    # predict

    predict = model.predict(x_t)
    valx[i, 0] = predict
    
    finpredict = scaler.inverse_transform(predict)
    finpredict = np.reshape(finpredict, (finpredict.shape[0]))
    finpredict = list(finpredict)
    predictions.append(finpredict)
    
predictions = np.array(predictions)    


# In[465]:


predictions = np.reshape(predictions, (predictions.shape[0]))
predictions


# In[346]:


# Write prediction to output csv
with open(outpath, 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
#     print(input_datalist[186:,1])
    
    tsmc = list(predictions)
    
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




