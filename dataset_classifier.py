# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:59:22 2020

@author: admin1
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
#what about using keras to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

#loading the data
path = os.path.join(os.getcwd(),'dataset','diabetes.csv')
df = pd.read_csv(path,delimiter=',')

X = df.iloc[:,:8]
y = df.iloc[:,8:]
y = np.array(y).reshape(-1) #chan

le = LabelEncoder()
y = le.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33)

#building model

model = Sequential()
model.add(Dense(128,activation='relu',input_dim=X.shape[1]))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#fitting the model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics =['acc'])
model.fit(X_train,y_train,epochs = 1000)

print(model.summary())
