# MLTG.py
# 2018 K. Fukami

## Machine-learning-based turbulence generator.
## Author: Kai Fukami (University of California, Los Angeles)

## Kai Fukami provides no guarantees for this code.  Use as-is and for academic research use only; no commercial use allowed without permission. For citations, please use the reference below:
# Ref: K. Fukami, Y. Nabae, K. Kawai, and K. Fukagata, "Synthetic turbulent inflow generator using machine learning," Phys. Rev. Fluids 4, 064603 (2019).
# The code is written for educational clarity and not for speed.

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Flatten, Reshape, LSTM
from keras.layers import CuDNNLSTM 
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


## Model part is only shown; data set needs to be prepared.
HIDDEN_SIZE = 3072

input_img = Input(shape=(96,256,4))
x = Conv2D(16, (3,3),activation='tanh', padding='same')(input_img)
x = Conv2D(16, (3,3),activation='tanh', padding='same')(x)
x = AveragePooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='tanh', padding='same')(x)
x = Conv2D(8, (3,3), activation='tanh', padding='same')(x)
x = AveragePooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='tanh', padding='same')(x)
x = Conv2D(8, (3,3), activation='tanh', padding='same')(x)
x = AveragePooling2D((2,2), padding='same')(x)
# MLP
x = Reshape([1,HIDDEN_SIZE])(x)
x = Dense(HIDDEN_SIZE,activation = 'tanh')(x)
x = Dense(HIDDEN_SIZE,activation = 'tanh')(x)
x = Reshape([12,32,8])(x) #3072
x = Conv2D(8, (3,3), activation='tanh', padding='same')(x)
x = Conv2D(8, (3,3), activation='tanh', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (3,3), activation='tanh', padding='same')(x)
x = Conv2D(8, (3,3), activation='tanh', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (3,3), activation='tanh', padding='same')(x)
x = Conv2D(16, (3,3), activation='tanh', padding='same')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(4, (3,3), padding='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')


from keras.callbacks import ModelCheckpoint,EarlyStopping
model_cb=ModelCheckpoint('./model.hdf5', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=100,verbose=1)
cb = [model_cb, early_cb]
history = autoencoder.fit(X_train,y_train,nb_epoch=5000,batch_size=100,verbose=1,callbacks=cb,shuffle=True,validation_data=[X_test, y_test])
import pandas as pd
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./history.csv',index=False)

