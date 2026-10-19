import numpy as np
import matplotlib.pyplot as plt
# This module will cover backpropagation in neural networks.

num_obs = 500
# Creates a matrix of random points in the 2D space [-1, 1] for each observation
x_mat_1 = np.random.uniform(-1, 1, (num_obs, 2))

# Adds a bias term (column of ones) to the input matrix
x_mat_b = np.ones((num_obs, 1))  # bias term

# Combines the original input matrix with the bias term to form the full input matrix
# This helps with including the bias term in the neural network computations.
# The resulting matrix has three columns: x1, x2, and the bias term.
x_mat_full = np.concatenate((x_mat_1, x_mat_b), axis=1)

#  Circle pattern
# Creates a 1-D array of labels based on whether the points are inside the circle of radius 0.75
y = (np.sqrt(x_mat_full[:,0]**2 + x_mat_full[:,1]**2)<.75).astype(int)

# Plots the points in the 2D space, coloring them based on their class labels
fig, ax = plt.subplots(figsize=(6, 6))
# Plots the points belonging to Class 0 in red
# It checks the y array to determine which points belong to Class 0
# and uses the 2 dimensions of x_mat_full for plotting
ax.plot(x_mat_full[y==0, 0], x_mat_full[y==0, 1], 'ro', label='Class 0')

# Plots the points belonging to Class 1 in blue
# It checks the y array to determine which points belong to Class 1
# and uses the 2 dimensions of x_mat_full for plotting
ax.plot(x_mat_full[y==1, 0], x_mat_full[y==1, 1], 'bo', label='Class 1')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Circle Pattern')
ax.legend()

def sigmoid(z):
    # Sigmoid activation function
    return 1 / (1 + np.exp(-z))

def loss_fn(y_true, y_pred, eps=1e-16):
    y_pred = np.maximum(y_pred, eps)
    y_pred = np.minimum(y_pred, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def forward_pass(w1, w2):
    # Forward pass through the network
    global x_mat
    global y
    global num_
    # First compute the new predictions 'y_pred' using the current weights
    z_2 = np.dot(x_mat, w1)
    a_2 = sigmoid(z_2)
    z_3 = np.dot(a_2, w2)
    y_pred = sigmoid(z_3)

    # Compute the gradient
    J_z_3_grad = -y + y_pred
    J_W_2_grad = np.dot(J_z_3_grad, a_2)
    a_2_z_2_grad = sigmoid(z_2)*(1-sigmoid(z_2))
    J_W_1_grad = (np.dot((J_z_3_grad).reshape(-1,1), W_2.reshape(-1,1).T)*a_2_z_2_grad).T.dot(x_mat).T
    gradient = (J_W_1_grad, J_W_2_grad)

    return y_pred, gradient
def plot_loss_accuracy(loss_vals, accuracies):
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Log Loss and Accuracy over iterations')
    
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(loss_vals)
    ax.grid(True)
    ax.set(xlabel='iterations', title='Log Loss')
    
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(accuracies)
    ax.grid(True)
    ax.set(xlabel='iterations', title='Accuracy')

np.random.seed(1241)

W_1 = np.random.uniform(-1,1,size=(3,4))
W_2 = np.random.uniform(-1,1,size=(4))
num_iter = 5000
learning_rate = .001
x_mat = x_mat_full


loss_vals, accuracies = [], []
for i in range(num_iter):
    ### Do a forward computation, and get the gradient
    y_pred, (J_W_1_grad, J_W_2_grad) = forward_pass(W_1, W_2)
    
    ## Update the weight matrices
    W_1 = W_1 - learning_rate*J_W_1_grad 
    W_2 = W_2 - learning_rate*J_W_2_grad
    
    ### Compute the loss and accuracy
    curr_loss = loss_fn(y,y_pred)
    loss_vals.append(curr_loss)
    acc = np.sum((y_pred>=.5) == y)/num_obs
    accuracies.append(acc)

    ## Print the loss and accuracy for every 200th iteration
    if((i%200) == 0):
        print('iteration {}, log loss is {:.4f}, accuracy is {}'.format(
            i, curr_loss, acc
        ))
plot_loss_accuracy(loss_vals, accuracies)

pred1 = (y_pred>=.5)
pred0 = (y_pred<.5)

fig, ax = plt.subplots(figsize=(8, 8))
# true predictions
ax.plot(x_mat[pred1 & (y==1),0],x_mat[pred1 & (y==1),1], 'ro', label='true positives')
ax.plot(x_mat[pred0 & (y==0),0],x_mat[pred0 & (y==0),1], 'bx', label='true negatives')
# false predictions
ax.plot(x_mat[pred1 & (y==0),0],x_mat[pred1 & (y==0),1], 'yx', label='false positives', markersize=15)
ax.plot(x_mat[pred0 & (y==1),0],x_mat[pred0 & (y==1),1], 'yo', label='false negatives', markersize=15, alpha=.6)
ax.set(title='Truth vs Prediction')
ax.legend(bbox_to_anchor=(1, 0.8), fancybox=True, shadow=True, fontsize='x-large');

plt.show()

