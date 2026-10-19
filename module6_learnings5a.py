import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

np.random.seed(1234)

num_obs = 100

x1 = np.random.uniform(0, 10, num_obs)
x2 = np.random.uniform(0, 10, num_obs)
const = np.ones(num_obs)
eps = np.random.normal(0, 0.5, num_obs)

b = 1.5
theta_1 = 2
theta_2 = 5

y = b + theta_1 * x1 + theta_2 * x2 + eps

x_mat = np.array([const, x1, x2]).T

from sklearn.linear_model import LinearRegression

lr_model = LinearRegression(fit_intercept=False)
lr_model.fit(x_mat, y)

# Print the estimated coefficients
print("Estimated coefficients:", lr_model.coef_)
print("True coefficients:", [b, theta_1, theta_2])

# Gradient Descent Exercise

# Initialize parameters for gradient descent
learning_rate = 0.01
num_iter = 1000
theta_initial = np.array([3,3,3,])

def gradient_descent(learning_rate, num_iter, theta_initial):
    theta = theta_initial.copy()
    theta_path = np.zeros((num_iter + 1, 3))
    theta_path[0, :] = theta_initial
    loss_vec = np.zeros(num_iter)

    for i in range(num_iter):
        y_pred = x_mat @ theta
        loss_vec[i] = np.sum((y - y_pred) ** 2)

        grad_vec = (x_mat.T @ (y_pred - y)) / num_obs
        theta = theta - learning_rate * grad_vec
        theta_path[i + 1, :] = theta

    return theta_path, loss_vec


def stochastic_gradient_descent(learning_rate, num_iter, theta_initial):
    theta = theta_initial.copy()
    theta_path = np.zeros((num_iter + 1, 3))
    theta_path[0, :] = theta_initial
    loss_vec = np.zeros(num_iter)

    for i in range(num_iter):
        idx = np.random.randint(0, num_obs)
        x_i = x_mat[idx, :]
        y_i = y[idx]

        y_pred_i = x_i.dot(theta)      # scalar
        loss_vec[i] = (y_i - y_pred_i) ** 2

        grad_vec = (y_pred_i - y_i) * x_i
        theta = theta - learning_rate * grad_vec
        theta_path[i + 1, :] = theta

    return theta_path, loss_vec


def mini_batch_gradient_descent(learning_rate, num_iter, theta_initial, batch_size):
    theta = theta_initial.copy()
    theta_path = np.zeros((num_iter + 1, 3))
    theta_path[0, :] = theta_initial
    loss_vec = np.zeros(num_iter)

    for i in range(num_iter):
        idx = np.random.randint(0, num_obs, batch_size)
        x_batch = x_mat[idx, :]
        y_batch = y[idx]

        y_pred_batch = x_batch @ theta
        loss_vec[i] = np.sum((y_batch - y_pred_batch) ** 2)

        grad_vec = (x_batch.T @ (y_pred_batch - y_batch)) / batch_size
        theta = theta - learning_rate * grad_vec
        theta_path[i + 1, :] = theta

    return theta_path, loss_vec

true_coef = [b, theta_1, theta_2]

def plot_ij(theta_path, i, j, ax):
    ax.plot(true_coef[i], true_coef[j],
            marker='p', markersize=15, label='true coef', 
            color='#778899')
    ax.plot(theta_path[:, i],theta_path[:, j],
            color='k', linestyle='--', marker='^', 
            markersize=5, markevery=50)
    ax.plot(theta_path[0, i], theta_path[0, j], marker='d', 
            markersize=15, label='start', color='#F08080')
    ax.plot(theta_path[-1, i], theta_path[-1, j], marker='o', 
            markersize=15, label='finish', color='#F08080')
    ax.set(
        xlabel='theta'+str(i),
        ylabel='theta'+str(j))
    ax.axis('equal')
    ax.grid(True)
    ax.legend(loc='best')
    

def plot_all(theta_path, loss_vec, learning_rate, num_iter, theta_initial, gdtype='Gradient Descent'):
    fig = plt.figure(figsize=(16, 16))
    title = '{gdtype} in the 3d parameter space - Learning rate is {lr} // {iters} iters // starting point {initial}'
    title = title.format(gdtype=gdtype, lr=learning_rate, 
                         iters=num_iter, initial=theta_initial)
    fig.suptitle(title, fontsize=20)
    ax = fig.add_subplot(2, 2, 1)
    plot_ij(theta_path, 0, 1, ax)
    ax = fig.add_subplot(2, 2, 2)
    plot_ij(theta_path, 0, 2, ax)
    ax = fig.add_subplot(2, 2, 3)
    plot_ij(theta_path, 1, 2, ax)
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(loss_vec)
    ax.set(xlabel='iterations', ylabel='squared loss')
    ax.grid(True)
    

theta_path, loss_vec = gradient_descent(learning_rate, num_iter, theta_initial)
plot_all(theta_path, loss_vec, learning_rate, num_iter, theta_initial, gdtype='Gradient Descent')
plt.show()

# Stochastic Gradient Descent Exercise

# Initialize parameters for stochastic gradient descent
learning_rate_sgd = 0.01
num_iter_sgd = 1000
theta_initial_sgd = np.zeros(3)

theta_path_sgd, loss_vec_sgd = stochastic_gradient_descent(learning_rate_sgd, num_iter_sgd, theta_initial_sgd)
plot_all(theta_path_sgd, loss_vec_sgd, learning_rate_sgd, num_iter_sgd, theta_initial_sgd, gdtype='Stochastic Gradient Descent')
plt.show()

# Mini-batch Gradient Descent Exercise

# Initialize parameters for mini-batch gradient descent
learning_rate_mbgd = 0.01
num_iter_mbgd = 1000
theta_initial_mbgd = np.zeros(3)
batch_size = 10

theta_path_mbgd, loss_vec_mbgd = mini_batch_gradient_descent(learning_rate_mbgd, num_iter_mbgd, theta_initial_mbgd, batch_size)
plot_all(theta_path_mbgd, loss_vec_mbgd, learning_rate_mbgd, num_iter_mbgd, theta_initial_mbgd, gdtype='Mini-batch Gradient Descent')
plt.show()

print("Estimated coefficients (Gradient Descent):", theta_path[-1])
print("Estimated coefficients (Stochastic Gradient Descent):", theta_path_sgd[-1])
print("Estimated coefficients (Mini-batch Gradient Descent):", theta_path_mbgd[-1])
    
