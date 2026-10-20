import pathlib
import numpy as np
import matplotlib.pyplot as plt

import PIL
from PIL import Image, ImageOps
import tensorflow as tf

from tensorflow import keras, squeeze
from tensorflow.keras import layers

import cv2

import warnings
warnings.warn = warnings.warn
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # tensorflow INFO and WARNING messages are not printed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disables oneDNN optimizations to avoid potential performance issues on some systems


data_path = "C:\\Users\\Lorenzo\\Desktop\\Python Projects\\Coursera Files\\flower_photos"
data_dir = pathlib.Path(data_path)

for folder in data_dir.iterdir():
    if folder.is_dir():
        print(f"This is the {folder.name} folder. It has {len(list(folder.glob('*.jpg')))} items.")

# Picture arrays
pics = list()
pics_arr = list()
p_class = list()

# Resizing the images to a uniform size for processing
img_height = 300
img_width = 300

# Iterating through each folder in the data directory to load and process images
for idx, folder in enumerate(data_dir.iterdir()):
    if folder.is_dir():
        for img_path in folder.glob('*.jpg'):
            img = Image.open(img_path).resize((img_width, img_height))
            pics.append(img)
            pics_arr.append(np.array(img))
            p_class.append(folder.name)

img = pics[0]

arr = np.array(img)
red_channel = arr[:, :, 0]
green_channel = arr[:, :, 1]
blue_channel = arr[:, :, 2]

channels = [red_channel, green_channel, blue_channel]
names = ['Red Channel', 'Green Channel', 'Blue Channel']

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(channels[i], cmap='gray')
    plt.title(names[i])
    plt.axis('off')
plt.show()

kernel = np.array([[-1,-1,-1],
                   [-1,8,-1],
                   [-1,-1,-1]]).reshape(3,3,1,1)
b = np.array([0.0])

model = keras.Sequential()
model.add(layers.Conv2D(input_shape = (img_height, img_width, 1),
                 filters=1, 
                 kernel_size=3, 
                 padding='same',
                 activation='relu'
                 ))
model.set_weights([kernel, b])
model.summary()

acts = []

for i in range(3):
    plt.subplot(1, 3, i + 1)

    input_ = channels[i].reshape((1, img_height, img_width, 1))
    act = model.predict(input_).squeeze(0).squeeze(2).astype('int').clip(0, 255)
    acts.append(act)

    plt.imshow(act, cmap='gray')
    plt.title(names[i])
    plt.axis('off')
plt.show()

arr_hat = np.stack(acts, axis=-1)
plt.imshow(arr_hat)
plt.axis('off')
plt.title('Reconstructed Image from Filtered Channels')
plt.show()