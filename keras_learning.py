# Keras Neural Network Learning Example
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# For building neural network models with Keras
import numpy as np
import pandas as pd

# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For splitting data into training and testing sets, scaling features, and evaluating model performance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

# Custom header names
names = ['times_pregnant', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age', 'has_diabetes']

# Load the diabetes dataset
diabetes_data = pd.read_csv('diabetes.csv', names=names, header=0)

# Print the shape of the dataset
print(diabetes_data.shape)

x = diabetes_data.drop('has_diabetes', axis=1)
y = diabetes_data['has_diabetes']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1111)

print(np.mean(y), np.mean(1-y))

# Test Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=200)
rf_model.fit(x_train, y_train)

# Make predictions on the test set - both "hard" predictions, and the scores (percent of trees voting yes)
y_pred_class_rf = rf_model.predict(x_test)
y_pred_prob_rf = rf_model.predict_proba(x_test)


print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred_class_rf)))
print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_rf[:,1])))

# Function to plot the ROC curve for a given model
def plot_roc(y_test, y_pred, model_name):
    fpr, tpr, thr = roc_curve(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, 'k-')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=.5)  # roc curve for random model
    ax.grid(True)
    ax.set(title='ROC Curve for {} on PIMA diabetes problem'.format(model_name),
           xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])
    plt.show()
plot_roc(y_test, y_pred_prob_rf[:, 1], 'RF')

# Test Keras Neural Network Model
normalizer = StandardScaler()
x_train_normalized = normalizer.fit_transform(x_train)
x_test_normalized = normalizer.transform(x_test)

# nn_model = Sequential()
# nn_model.add(Dense(12, input_shape= (8,), activation='sigmoid'))
# nn_model.add(Dense(1, activation='sigmoid'))

# nn_model.summary()

# # Train the Keras Neural Network Model
# nn_model.compile(SGD(learning_rate=0.003), loss='binary_crossentropy', metrics=['accuracy'])
# run_history = nn_model.fit(x_train_normalized, y_train, epochs=200, validation_data=(x_test_normalized, y_test))

# y_pred_prob_nn = nn_model.predict(x_test_normalized)
# y_pred_class_nn = (y_pred_prob_nn > 0.5).astype(int)

# print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred_class_nn)))
# print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_nn)))
# plot_roc(y_test, y_pred_prob_nn, 'NN')

# # Run history of the Keras Neural Network Model
# print(run_history.history.keys())

# fig, ax = plt.subplots(figsize=(8, 8))
# # plot loss
# ax.plot(run_history.history['loss'], label='train loss')
# ax.plot(run_history.history['val_loss'], label='val loss')
# ax.set(title='Loss over epochs', xlabel='Epoch', ylabel='Loss')
# ax.legend()

# # plot accuracy
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.plot(run_history.history['accuracy'], label='train accuracy')
# ax.plot(run_history.history['val_accuracy'], label='val accuracy')
# ax.set(title='Accuracy over epochs', xlabel='Epoch', ylabel='Accuracy')
# ax.legend()
# plt.show()

# Test another Keras Neural Network Model with different architecture
model_2 = Sequential()
model_2.add(Dense(6, input_shape=(8,), activation="relu"))
model_2.add(Dense(6,  activation="relu"))
model_2.add(Dense(6,  activation="relu"))
model_2.add(Dense(6,  activation="relu"))
model_2.add(Dense(6,  activation="relu"))
model_2.add(Dense(1, activation="sigmoid"))

model_2.compile(SGD(learning_rate=0.003), "binary_crossentropy", metrics=["accuracy"])
run_hist_2 = model_2.fit(x_train_normalized, y_train, validation_data=(x_test_normalized, y_test), epochs=325)

run_hist_2.history.keys()

n = len(run_hist_2.history["loss"])

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1)
ax.plot(range(n), (run_hist_2.history["loss"]),'r.', label="Train Loss")
ax.plot(range(n), (run_hist_2.history["val_loss"]),'b.', label="Validation Loss")
ax.legend()
ax.set_title('Loss over iterations')

ax = fig.add_subplot(1, 2, 2)
ax.plot(range(n), (run_hist_2.history["accuracy"]),'r.', label="Train Acc")
ax.plot(range(n), (run_hist_2.history["val_accuracy"]),'b.', label="Validation Acc")
ax.legend(loc='lower right')
ax.set_title('Accuracy over iterations')
plt.show()