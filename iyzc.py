# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Used Library
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#train and test data
data1 = pd.read_csv("train.csv")
data2 = pd.read_csv("test.csv") 
data3= pd.read_csv("SampleSubmission.csv")

#Data normalization to 0-1
scaler = MinMaxScaler(feature_range=(0, 1))

data2=data2.drop(columns="ID")
data2 = scaler.fit_transform(data2)
data1 = scaler.fit_transform(data1)

#Model input and output
inp = data1[:,0:12]
out = data1[:,12]

#Creating Model with 64-128-256 dense, 12 dimmension, OLS method and train %92 to predict %8 then give accuracy rate
model = Sequential()
model.add(Dense(64, input_dim=12))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy',metrics =['accuracy'])
model.fit(inp,out, epochs=5, batch_size=32, validation_split = 0.08)


#predict spesific sample(must print 1)
predict = np.array([1.0114220195155382e-05,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.768272042834465,0.25806451612903225,0.03075531433740389]).reshape(1,12)
print(model.predict_classes(predict))

#predict whole test sample
predict2=model.predict_classes(data2)
predict2 = pd.DataFrame(predict2, columns=["ISFRAUD"])

#Count for given IsFraud=1 (must be %1~ of whole sample)
predict2[(predict2["ISFRAUD"]==1)].count()

#creating ID and merge predict2
ID = pd.Series(np.linspace(0,39999,40000)) 
ID= pd.DataFrame(ID.astype(int))
SampleSubmission = ID.join(predict2, how="outer")
SampleSubmission.columns=["ID", "ISFRAUD"]

#Writing csv 
SampleSubmission.to_csv(r'C:\Users\fdc\SampleSubmission.csv')