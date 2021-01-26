import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler

df = pd.read_csv('NYW.csv')
L = len(df)
Y = np.array([df.iloc[:,3]])
plt.plot(Y[0,:])
plt.savefig("fig1.png")

X1 = Y[:,0:L-5]
X2 = Y[:,0:L-4]
X3 = Y[:,0:L-3]
X = np.concatenate([X1,X2,X3],axis=0)
#X = np.concatenate((X1,X2,X3[:,None]),axis=1)
X = np.transpose(X)
Y = np.transpose(Y[:,3:L-2])
sc= MinMaxScaler()
sc.fit(X)
X= sc.transform(X)
sc1 = MinMaxScaler()
sc1.fit(Y)
Y= sc.transform(Y)
X= np.reshape(X,(X.shape[0],1,X.shape[1]))
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

model = Sequential()
model.add(LSTM(10,activation='tanh',input_shape=(1,3),recurrent_activation='hard_sigmoid'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='rmsprop',metrics=[metrics.mae])
model.fit(X_train,Y_train,epochs= 25,verbose=2)
predict= model.predict(X_test)

plt.figure(2)
plt.scatter(Y_test,predict)
plt.show()
plt.figure(3)
Real=plt.plot(Y_test)
Predict= plt.plot(predict)
plt.legend([Predict,Real],["Predicted Data","Real Data"])
plt.savefig("fig2")
