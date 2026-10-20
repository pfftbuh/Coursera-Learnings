import pathlib
import numpy as np
import matplotlib.pyplot as plt

import PIL
from PIL import Image, ImageOps
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

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

images = []
for i in range(5):
    images.append(pics_arr[i])

    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i])
    plt.axis('off')

# Average vs Max Pooling
max_pool = tf.keras.Sequential([tf.keras.layers.MaxPooling2D(pool_size=(2, 2))])
avg_pool = tf.keras.Sequential([tf.keras.layers.AveragePooling2D(pool_size=(2, 2))])

figs1, axes1 = plt.subplots(1, 5, constrained_layout=True)
figs2, axes2 = plt.subplots(1, 5, constrained_layout=True)
figs1.suptitle('Max Pooling')
figs2.suptitle('Average Pooling')

for img, ax1, ax2 in zip(images, axes1, axes2):
    img = img.reshape((1, img_height, img_width, 3))

    ax1.imshow(max_pool.predict(img).squeeze(0) / 255.0)
    ax1.axis('off')

    ax2.imshow(avg_pool.predict(img).squeeze(0) / 255.0)
    ax2.axis('off')

plt.show()

# Train a simple CNN model on the CIFAR 10 dataset
def load_cifar10_data():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

(X_train, y_train), (X_test, y_test) = load_cifar10_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Set convolutional layer parameters
conv = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')


max_pool = layers.MaxPooling2D(pool_size=(2, 2))

# Create a simple CNN model

model = Sequential()

model.add(conv)
model.add(max_pool)

# Double the number of filters in the next convolutional layer
conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_initializer='he_uniform')
conv2b = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                       kernel_initializer='he_uniform')
model.add(conv2)
model.add(conv2b)

conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_initializer='he_uniform')
conv3b = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                       kernel_initializer='he_uniform')
model.add(conv3)
model.add(conv3b)

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Dense(len(class_names), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Test model with sample images from dataset. Show predicted and actual class names for each image
for i in range(5):
    img = X_test[i]
    img_expanded = np.expand_dims(img, axis=0)
    prediction = model.predict(img_expanded)
    predicted_class = class_names[np.argmax(prediction)]
    actual_class = class_names[np.argmax(y_test[i])]
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}\nActual: {actual_class}")

plt.show()