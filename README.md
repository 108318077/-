# -import pandas as pd<br>
import numpy as np<br>
from keras.models import Sequential<br>
from keras import optimizers<br>
from keras.layers import *<br>
from keras.callbacks import *<br>
from sklearn.preprocessing import *<br>
<br>
df_train = pd.read_csv('train-v3.csv') <br>
df_valid = pd.read_csv('valid-v3.csv') <br>
df_test = pd.read_csv('test-v3.csv')<br>

data_title=['sale_yr','sale_month','sale_day','bedrooms','bathrooms','floors','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']<br>
<br>
df_train.drop(['id'],inplace = True,axis=1) <br>
df_valid.drop(['id'],inplace = True,axis=1) <br>
df_test.drop(['id'],inplace = True,axis=1)<br>

Y_train = df_train["price"].values<br>
X_train = df_train[data_title].values<br>
Y_valid = df_valid["price"].values<br>
X_valid = df_valid[data_title].values<br>
X_test = df_test[data_title].values<br>

X_train_normal = scale(X_train)<br>
X_valid_normal = scale(X_valid)<br>
X_test_normal = scale(X_test)<br>


model = Sequential()<br>
model.add(Dense(1000, input_dim=X_train.shape[1],kernel_initializer='normal',activation='relu'))<br>
model.add(Dense(1000, kernel_initializer='normal',activation='relu'))<br>
model.add(Dense(1, kernel_initializer='normal'))<br>

model.compile(loss='MAE', optimizer='adam')<br>

nb_epoch = 500<br>
batch_size = 32<br>

file_name = str(nb_epoch)+'_'+str(batch_size)<br>
TB = TensorBoard(log_dir='logs/'+ file_name, histogram_freq = 0)<br>
model.fit(X_train_normal, Y_train, batch_size = batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_valid_normal, Y_valid),callbacks=[TB])<br>
model.save('h5/'+file_name+'.h5')<br>
Y_predict = model.predict(X_test_normal)<br>
np.savetxt('Output_16.csv', Y_predict, delimiter = ',')<br>
