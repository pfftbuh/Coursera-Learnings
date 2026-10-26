import os
import copy
import skillsnetwork

import numpy as np
from numpy.core.fromnumeric import reshape

import tensorflow as tf
    
import keras
from keras import layers,Input,Sequential 
from keras.layers import Dense,Flatten,Reshape,Conv2DTranspose,Conv2D
from keras.models import Model


from keras.layers import Conv2D
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from keras.datasets import mnist

def graph_history(history, title='Log Loss and Accuracy over iterations'):

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(title)
    N_plots=len(history.history.keys())
    color_list=['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w','bx','rx']
    for i,(key, items) in enumerate(history.history.items()):
        ax = fig.add_subplot(1, N_plots, i+1)
        ax.plot(items,c=color_list[i])
        ax.grid(True)
        ax.set(xlabel='iterations', title=key)

    plt.show()

def plot_images(top,bottom,start=0,stop=5,reshape_x=(28,28),reshape_xhat=(28,28)):
    
    '''
    this function plots images from the start index to the stop index from two datasets
    
    '''

    n_samples=stop-start

    for i,img_index in enumerate(range(start,stop)):
        
        # Display original
        ax = plt.subplot(2, n_samples, i + 1)
        plt.imshow(top[img_index].reshape(reshape_x[0], reshape_x[1]), cmap="gray")

        if i==n_samples//2:
            plt.title("original images")

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.imshow(bottom[img_index].reshape(reshape_xhat[0], reshape_xhat[1]), cmap="gray")


        if i==n_samples//2:
            plt.title("encoded images")

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.show()

(xtrain,ytrain),(xtest,ytest) = keras.datasets.fashion_mnist.load_data()

xtrain = xtrain.astype('float32')/255
xtest = xtest.astype('float32')/255

xtrain = xtrain.reshape((len(xtrain), np.prod(xtrain.shape[1:])))
xtest = xtest.reshape((len(xtest), np.prod(xtest.shape[1:])))

print(xtrain.shape)
print(xtest.shape)

input_image=Input(shape=(784,))

# encoding_dim = 36
# encoder = Dense (encoding_dim, activation='relu')(input_image)

# decoder = Dense(784, activation='sigmoid')(encoder)

# autoencoder = Model(input_image, decoder)

# autoencoder.compile(optimizer='adam',loss='binary_crossentropy')

# history=autoencoder.fit(xtrain,xtrain,epochs=25,batch_size=256,shuffle=True,validation_data=(xtest,xtest))

# graph_history(history=history, title='Log Loss and Accuracy over iterations')

# xhat=autoencoder.predict(xtest)

# plot_images(xtest,xhat,start=0,stop=5)

class Deep_Autoencoder(Model):
    def __init__(self, latent_dim_1,latent_dim_2):
        super(Deep_Autoencoder, self).__init__()
        self.latent_dim_1 = latent_dim_1 
        self.latent_dim_2 = latent_dim_2
        self.encoder = Sequential([layers.Flatten(),Dense(latent_dim_1,activation='relu'),Dense(latent_dim_2,activation='relu')])
        self.decoder = Sequential([Dense(latent_dim_1,activation='relu'),Dense(784,activation='sigmoid'),Reshape((28,28))])
    def call(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

latent_dim_1 =128
latent_dim_2=3
deep_autoencoder=Deep_Autoencoder(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2)

xtrain_images = xtrain.reshape((-1, 28, 28))
xtest_images = xtest.reshape((-1, 28, 28))

deep_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history=deep_autoencoder.fit(xtrain_images,xtrain_images,epochs=50,batch_size=256,shuffle=True,validation_data=(xtest_images,xtest_images))


xhat=deep_autoencoder.predict(xtest_images)
plot_images(xtest, xhat, start=0, stop=5)

class CNN_Autoencoder(Model):
    def __init__(self):
        super(CNN_Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            Conv2D(16, (3, 3), activation='relu', padding='same', strides=1),
            Conv2D(8, (3, 3), activation='relu', padding='same', strides=1)])

        self.decoder = tf.keras.Sequential([
            Conv2DTranspose(8, kernel_size=3, strides=1, activation='relu', padding='same'),
            Conv2DTranspose(16, kernel_size=3, strides=1, activation='relu', padding='same'),
            Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

cnn_autoencoder=CNN_Autoencoder()

cnn_autoencoder.compile(optimizer='adam',  loss='mse')
xtrain_cnn = xtrain.reshape((-1, 28, 28, 1))
xtest_cnn = xtest.reshape((-1, 28, 28, 1))
history=cnn_autoencoder.fit(
    xtrain_cnn,
    xtrain_cnn,
    epochs=10,
    validation_data=(xtest_cnn, xtest_cnn)
)

graph_history(history, title='Log Loss and Accuracy over iterations')
xhat=deep_autoencoder.predict(xtest_images)
plot_images(xtest, xhat, start=0, stop=5)