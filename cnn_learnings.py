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


data_path = "C:\\Users\\Lorenzo\\Desktop\\Python Projects\\Coursera Files\\flower_photos"
data_dir = pathlib.Path(data_path)

for folder in data_dir.iterdir():
    if folder.is_dir():
        print(f"This is the {folder.name} folder. It has {len(list(folder.glob('*.jpg')))} items.")

# Convolutional Neural Network Testing

