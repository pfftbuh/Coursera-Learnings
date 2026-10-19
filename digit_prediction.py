import joblib
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the test set
digits = pd.read_csv('digits.csv')
labels = digits['label']
digits = np.array(digits.drop('label', axis=1)).astype(np.float32)
digits /= 255.0
split_idx = int(0.7 * len(digits))
X_test = digits[split_idx:]
y_test = labels[split_idx:]


# Load the best model (for future use)
best_mlp_loaded = joblib.load('best_mlp_model.pkl')
print("Best MLP model loaded from 'best_mlp_model.pkl'")

# Choose a random number from the test set and predict its label. Show all in one plot
for i in range(5):
    random_idx = random.randint(0, len(X_test) - 1)
    random_sample = X_test[random_idx].reshape(1, -1)
    predicted_label = best_mlp_loaded.predict(random_sample)[0]
    true_label = y_test.iloc[random_idx]
    print(f"Random test sample index: {random_idx}")
    print(f"Predicted label: {predicted_label}")
    print(f"True label: {true_label}")
    print('-'*30)
    plt.imshow(random_sample.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_label}, True: {true_label}")
    plt.show()
    plt.close()