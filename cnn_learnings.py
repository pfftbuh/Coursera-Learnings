import pathlib
import numpy as np
import matplotlib.pyplot as plt

import PIL
from PIL import Image, ImageOps
import tensorflow as tf

from tensorflow import keras
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
            img = Image.open(img_path)
            img = ImageOps.fit(img, (img_width, img_height), Image.Resampling.LANCZOS)
            pics.append(img)
            pics_arr.append(np.array(img))
            p_class.append(folder.name)

img = pics[0]

# Shows the first image in the dataset
# plt.imshow(img)
# plt.show()

# Convolutional Neural Network Testing

# Sobel filter gradients for edge detection

def sobel(img, strides, padding, activation=None):

    # Initialize the model
    input_shape = (img_height, img_width, 1) # Grayscale images have one channel

    input_layer = layers.Input(shape=input_shape)

    h_conv = layers.Conv2D(filters=1, kernel_size=3, padding='valid', kernel_initializer=h_grad, strides=1)
    v_conv = layers.Conv2D(filters=1, kernel_size=3, padding='valid', kernel_initializer=v_grad, strides=1)

    h_model = keras.Sequential([input_layer, h_conv])
    v_model = keras.Sequential([input_layer, v_conv])

    gray_img = ImageOps.grayscale(img)
    gray_img_arr = np.array(gray_img).reshape((1, img_height, img_width, 1)) # Reshape to match the input shape of the model

    # Predict the gradients
    output_h = h_model.predict(gray_img_arr)
    output_v = v_model.predict(gray_img_arr)

    # Compute the gradient magnitude
    G = np.sqrt(np.add(np.multiply(output_h, output_h), np.multiply(output_v, output_v)))

    return G[0, :, :, 0] # Return the gradient magnitude for the first (and only) image in the batch

def v_grad(shape, dtype=None):
    grad = np.array(
        [
            [1,0,-1],
            [2,0,-2],
            [1,0,-1]
        ]
    ).reshape((3,3,1,1)) # this line is quite important, we are saying we want one 3x3 kernel each for one channel of pixels (grayscale)

    assert grad.shape == shape, f"Expected shape {shape}, but got {grad.shape}"
    return keras.backend.variable(grad, dtype='float32')

def h_grad(shape, dtype=None):
    grad = np.array(
        [
            [1,2,1],
            [0,0,0],
            [-1,-2,-1]
        ]
    ).reshape((3,3,1,1)) # this line is quite important, we are saying we want one 3x3 kernel each for one channel of pixels (grayscale)

    assert grad.shape == shape, f"Expected shape {shape}, but got {grad.shape}"
    return keras.backend.variable(grad, dtype='float32')

fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
for i, img in enumerate(pics[:3]):
    axs[i].imshow(sobel(img, strides=1, padding='same'), cmap='gray')
    axs[i].set_title(f"Image {i+1} Sobel Filtered")
    axs[i].axis('off')

# Show the filtered images
plt.show()
