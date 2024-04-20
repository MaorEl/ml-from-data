###### Your ID ######
# ID1: 312412927
# ID2: 204226815
#####################

# imports 
import numpy as np
import pandas as pd


def preprocess(X,y):
    X = mean_normalization(X)
    y = mean_normalization(y)

    return X, y


def mean_normalization(matrix):
    min_vector = np.min(matrix, axis=0)
    max_vector = np.max(matrix, axis=0)
    mean_vector = np.mean(matrix, axis=0)
 
    matrix = matrix-mean_vector
    matrix = matrix/(max_vector-min_vector)
    return matrix


def apply_bias_trick(X):
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    bias_column =  np.ones((X.shape[0], 1))
    X_bias = np.hstack((bias_column, X))

    return X_bias

# n - number of  instances(row - 0)
# m - number of features(col -1)
def compute_cost(X, y, theta):
    number_of_instances = X.shape[0]
    number_of_features = X.shape[1]

    if len(theta) != number_of_features:
        raise Exception("Theta must be a vector with a size equal to the number of features.")
    if len(y) != number_of_instances:
        raise Exception("True labels must be a vector with a size equal to the number of instances.")

    h_teta = X @ theta
    sqr_errors = np.square(h_teta - y )
    
    return  1 / (2 * number_of_instances) * np.sum(sqr_errors)


def gradient_descent(X, y, theta, alpha, num_iters):
    number_of_instances = X.shape[0]
    J_history = np.empty(num_iters)
    theta_copy = np.copy(theta)
    for iteration in range(num_iters):
        theta_copy =generate_new_theta(X, y, alpha, number_of_instances, theta_copy)
        J_history[iteration] = compute_cost(X, y, theta_copy)
    return theta_copy, J_history

def compute_pinv(X, y):
    pinv_X = (np.linalg.inv(X.T @ X)) @ X.T 
    pinv_theta = pinv_X @ y
    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    number_of_instances = X.shape[0]
    J_history = np.empty(num_iters)
    theta_copy = np.copy(theta)
    prev_j = np.inf

    for iteration in range(num_iters):
        J_history[iteration] = compute_cost(X, y, theta_copy)
        theta_copy = generate_new_theta(X, y, alpha, number_of_instances, theta_copy)
        if prev_j - J_history[iteration] < 1e-8:
            break
        prev_j = J_history[iteration]

    return theta_copy, J_history


def generate_new_theta(X, y, alpha, number_of_instances, theta_copy):
    h_teta = X @ theta_copy
    errors = h_teta - y
    theta_copy -= (alpha/number_of_instances) * (errors @ X)
    return theta_copy

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    number_of_features = X_train.shape[1]
    theta = np.random.random(size=number_of_features)
    alpha_cost_lambda = lambda  alpha1 : compute_cost(X_val,y_val, efficient_gradient_descent(X_train, y_train, theta, alpha1, iterations)[0])
    alpha_dict = { alpha : alpha_cost_lambda(alpha) for alpha in alphas}
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    selected_features = []
    X_train = apply_bias_trick(X_train)
    X_val = apply_bias_trick(X_val)
    number_of_features = X_train.shape[1]

    for _ in range(5):
        best_cost = np.inf
        best_feature = None

        for feature in range(1, number_of_features):  # skipping the bias which is located on index 0
            if feature-1 in selected_features:
                continue

            selected_features.append(feature)
            x_train_to_use = X_train[:, [0] + selected_features]
            theta = np.random.random(size=x_train_to_use.shape[1])

            x_val_to_use = X_val[:, [0] + selected_features]
            updated_theta = efficient_gradient_descent(x_train_to_use, y_train, theta, best_alpha, iterations)[0]
            cost = compute_cost(x_val_to_use, y_val, updated_theta)

            if cost < best_cost:
                best_cost = cost
                best_feature = feature

            selected_features.pop()

        if best_feature is not None:
            selected_features.append(best_feature-1)

    return selected_features

def create_square_features(df):
    df_poly = df.copy()

    # creating the square features
    for column in df.columns:
        new_column = f"{column}^2"
        df_poly[new_column] = df[column] ** 2

    # creating the cross features
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            new_column = f"{df.columns[i]}*{df.columns[j]}"
            df_poly[new_column] = df[df.columns[i]] * df[df.columns[j]]

    return df_poly