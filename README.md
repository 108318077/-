# -import pandas as pd
import numpy as np
from keras.models import Sequential
from keras import optimizers
from keras.layers import *
from keras.callbacks import *
from sklearn.preprocessing import *

df_train = pd.read_csv('train-v3.csv') 
df_valid = pd.read_csv('valid-v3.csv') 
df_test = pd.read_csv('test-v3.csv')

data_title=['sale_yr','sale_month','sale_day','bedrooms','bathrooms','floors','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']

df_train.drop(['id'],inplace = True,axis=1) 
df_valid.drop(['id'],inplace = True,axis=1) 
df_test.drop(['id'],inplace = True,axis=1)

Y_train = df_train["price"].values
X_train = df_train[data_title].values
Y_valid = df_valid["price"].values
X_valid = df_valid[data_title].values
X_test = df_test[data_title].values

X_train_normal = scale(X_train)
X_valid_normal = scale(X_valid)
X_test_normal = scale(X_test)


model = Sequential()
model.add(Dense(1000, input_dim=X_train.shape[1],kernel_initializer='normal',activation='relu'))
model.add(Dense(1000, kernel_initializer='normal',activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

model.compile(loss='MAE', optimizer='adam')

nb_epoch = 500
batch_size = 32

file_name = str(nb_epoch)+'_'+str(batch_size)
TB = TensorBoard(log_dir='logs/'+ file_name, histogram_freq = 0)
model.fit(X_train_normal, Y_train, batch_size = batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_valid_normal, Y_valid),callbacks=[TB])
model.save('h5/'+file_name+'.h5')
Y_predict = model.predict(X_test_normal)
np.savetxt('Output_16.csv', Y_predict, delimiter = ',')
