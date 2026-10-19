import warnings
warnings.simplefilter('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN optimizations

import numpy as np
import pandas as pd
from itertools import accumulate
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics
from sklearn.metrics import mean_squared_error

sns.set_context('notebook')
sns.set_style('white')

raw_data = pd.read_csv('day.csv')
print(raw_data.shape)
print(raw_data.head())

clean_data = raw_data.drop(['instant', 'dteday','casual','registered'], axis=1)
print(clean_data.shape)
print(clean_data.head())

# Visualize the relationships between each feature and the target variable 'cnt'
sns.pairplot(clean_data, x_vars=clean_data.columns[:-1], y_vars='cnt', height=2.5, aspect=1)
plt.show()

# Split the data into features (X) and target (y) for modeling
X = clean_data.drop(['cnt'], axis=1)
y = clean_data['cnt']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features using Normalization layer
normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(X_train_df))

# This prepares the normalized training and testing feature arrays for modeling
X_train = normalizer(np.array(X_train_df))
X_test = normalizer(np.array(X_test_df))




# Build a simple linear regression model using Keras
temp_arr = np.array(X_train_df['temp'])

temp_normalized = layers.Normalization(input_shape=[1,], axis=None)
temp_normalized.adapt(temp_arr)

temp_model = keras.Sequential([
    layers.Dense(1, input_shape=(1,))
])

temp_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the model
history = temp_model.fit(temp_normalized(np.array(X_train_df['temp'])), y_train, validation_split=0.2, epochs=200)

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Test the model on the test set
test_loss, test_mae = temp_model.evaluate(temp_normalized(np.array(X_test_df['temp'])), y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Output 5 predictions from the test set and its corresponding true values
predictions = temp_model.predict(temp_normalized(np.array(X_test_df['temp'])))
print("Predictions:", predictions[:5])
print("True Values:", y_test[:5].values)

# Calculate and print the mean squared error for the test set
mse = mean_squared_error(y_test, predictions)
print(f"Test MSE: {mse}")




