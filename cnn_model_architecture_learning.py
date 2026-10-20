import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
print(tf.__version__)
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import train_test_split

seed = 7
np.random.seed(seed)


# Functional API
df = pd.read_csv(r"C:\Users\Lorenzo\Desktop\Python Projects\Coursera Files\sonar.csv", header=None)

dataset = df.values

X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

# Encode class values as integers
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X, encoded_Y, test_size=0.2, random_state=seed)


def functional_model():
    inputs = tf.keras.Input(shape=(60,))
    x = layers.Dense(60, activation='relu')(inputs)
    x = layers.Dense(30, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Functional API with Batch Normalization
def functional_model_with_batch_normalization():
    inputs = tf.keras.Input(shape=(60,))
    x = layers.Dense(60, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(30, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Functional API with Dropout
def functional_model_with_dropout():
    inputs = tf.keras.Input(shape=(60,))
    x = layers.Dense(60, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(30, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Functional API with L1 Regularization
def functional_model_with_l1_regularization():
    inputs = tf.keras.Input(shape=(60,))
    x = layers.Dense(60, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01))(inputs)
    x = layers.Dense(30, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01))(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Functional API with L2 Regularization
def functional_model_with_l2_regularization():
    inputs = tf.keras.Input(shape=(60,))
    x = layers.Dense(60, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = layers.Dense(30, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

model1 = functional_model()
model1.summary()
model1.fit(X_train, y_train, epochs=100, batch_size=5, shuffle=True, verbose=0)

score = model1.evaluate(X_test, y_test, verbose=0)

print("Functional API Model Test loss:", score[0])
print("Functional API Model Test accuracy:", score[1])

model2 = functional_model_with_batch_normalization()
model2.summary()
model2.fit(X_train, y_train, epochs=100, batch_size=5, shuffle=True, verbose=0)

score = model2.evaluate(X_test, y_test, verbose=0)

print("Functional API Model with Batch Normalization Test loss:", score[0])
print("Functional API Model with Batch Normalization Test accuracy:", score[1])

model3 = functional_model_with_dropout()
model3.summary()
model3.fit(X_train, y_train, epochs=100, batch_size=5, shuffle=True, verbose=0)

score = model3.evaluate(X_test, y_test, verbose=0)
print("Functional API Model with Dropout Test loss:", score[0])
print("Functional API Model with Dropout Test accuracy:", score[1])

model4 = functional_model_with_l1_regularization()
model4.summary()
model4.fit(X_train, y_train, epochs=100, batch_size=5, shuffle=True, verbose=0)

score = model4.evaluate(X_test, y_test, verbose=0)
print("Functional API Model with L1 Regularization Test loss:", score[0])
print("Functional API Model with L1 Regularization Test accuracy:", score[1])

model5 = functional_model_with_l2_regularization()
model5.summary()
model5.fit(X_train, y_train, epochs=100, batch_size=5, shuffle=True, verbose=0)

score = model5.evaluate(X_test, y_test, verbose=0)
print("Functional API Model with L2 Regularization Test loss:", score[0])
print("Functional API Model with L2 Regularization Test accuracy:", score[1])


