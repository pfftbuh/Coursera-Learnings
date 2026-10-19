# Module 6 Learnings
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

digits = pd.read_csv('digits.csv')

labels = digits['label']

digits = np.array(digits.drop('label', axis=1)).astype(np.float32)
digits.shape, labels.shape

print(digits.shape, labels.shape)

split = 0.7, 0.3

digits /= 255.0

split_idx = int(split[0] * len(digits))
X_train, X_test = digits[:split_idx], digits[split_idx:]
y_train, y_test = labels[:split_idx], labels[split_idx:]

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

mlp = MLPClassifier().fit(X_train, y_train)

y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy: %f" % accuracy)
print(classification_report(y_test, y_pred))

parameters = {'hidden_layer_sizes':[50, 100, 200],
              'alpha': [0.001, 0.01, 0.1], 
              'max_iter': [200, 500, 800], 
              'learning_rate_init':[0.0001, 0.001, 0.01, 0.1]}

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

mlp = MLPClassifier()

random_search = RandomizedSearchCV(estimator=mlp, param_distributions=parameters, cv=5)
# Use smaller sample for quicker testing during development
X_train_sample = X_train[:3000]
y_train_sample = y_train[:3000]
random_search.fit(X_train_sample, y_train_sample)

best_mlp = random_search.best_estimator_
y_pred = best_mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy after RandomizedSearchCV: %f" % accuracy)
print(classification_report(y_test, y_pred))

# Save the best model
import joblib
joblib.dump(best_mlp, 'best_mlp_model.pkl')
print("Best MLP model saved as 'best_mlp_model.pkl'")
