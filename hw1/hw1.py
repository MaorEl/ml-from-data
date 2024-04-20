###### Your ID ######
# ID1: 204226815
# ID2: 987654321 Todo Tzachi - change to your ID
#####################

# imports 
import numpy as np
import pandas as pd

'''def preprocess(X,y):
    print(X.ndim)
    print(y.ndim)
    if X.ndim == 1:
        # Handle the case where X is a one-dimensional array
        X = mean_normalization(X)
    elif X.ndim == 2:
        # Handle the case where X is a two-dimensional array
        number_of_features = X.shape[1]
        for i in range(number_of_features):
            normal_feature = mean_normalization(X[:, i])
            X[:, i] = normal_feature
    else:
        raise ValueError("X must be either a 1D or 2D array")

    y = mean_normalization(y)

    return X, y

def mean_normalization():
    feature_min = feature.min()
    feature_max = feature.max()
    feature_mean = feature.mean()

    feature = feature-feature_mean
    feature = feature/(feature_max-feature_min)
    return feature'''

def preprocess(X,y):
    print(f"x.dim: {X.ndim}")
    print(f"y.dim: {y.ndim}")
    X = mean_normalization(X)
    y = mean_normalization(y)

    return X, y

def mean_normalization(matrix):
    min_vector = np.min(matrix, axis=0)
    max_vector = np.max(matrix, axis=0)
    mean_vector = np.mean(matrix, axis=0)
    print(f"min vector: {min_vector}")
    print(f"max vector: {max_vector}")
    print(f"mean vector: {mean_vector}")
 
    matrix = matrix-mean_vector
    matrix = matrix/(max_vector-min_vector)
    return matrix

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
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

    h_theta = X @ theta
    squared_errors = np.square(h_theta - y)
    
    return 1 / (2 * number_of_instances) * np.sum(squared_errors)


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using
    the training set. Gradient descent is an optimization algorithm
    used to minimize some (loss) function by iteratively moving in
    the direction of steepest descent as defined by the negative of
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    number_of_instances = X.shape[0]
    J_history = np.empty(num_iters)
    theta_copy = np.copy(theta)
    for iteration in range(num_iters):
        theta_copy = generate_new_theta(X, y, alpha, number_of_instances, theta_copy)
        J_history[iteration] = compute_cost(X, y, theta_copy)

    return theta_copy, J_history




def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_X = (np.linalg.inv(X.T @ X)) @ X.T 
    pinv_theta = pinv_X @ y

    print(pinv_theta)
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    number_of_instances = X.shape[0]
    J_history = np.empty(num_iters)
    theta_copy = np.copy(theta)
    prev_j = compute_cost(X, y, theta_copy)
    for iteration in range(num_iters):
        theta_copy = generate_new_theta(X, y, alpha, number_of_instances, theta_copy)
        J_history[iteration] = compute_cost(X, y, theta_copy)
        if prev_j - J_history[iteration] < 1e-8:
            print(f"for alpha: {alpha} breaking in iteration: {iteration}. when theta: {theta_copy}. origin theta:{theta} when cost change: {prev_j - J_history[iteration]}")
            break
        prev_j = J_history[iteration]

    return theta_copy, J_history


def generate_new_theta(X, y, alpha, number_of_instances, theta):
    h_theta = X @ theta
    errors = h_theta - y
    theta -= (alpha / number_of_instances) * (X.T @ errors) # todo maor - why X.T?

    return theta

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    number_of_features = X_train.shape[1]
    theta = np.random.random(size=number_of_features)
    alpha_cost_lambda = lambda  alpha1 : compute_cost(X_val,y_val, efficient_gradient_descent(X_train, y_train, theta, alpha1, iterations)[0])
    alpha_dict = { alpha : alpha_cost_lambda(alpha) for alpha in alphas}
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    # TODO: Implement the function and find the best alpha value.             #
    '''
    Adding additional features to our regression model makes it more complicated but does not necessarily improves performance. 
    
    Forward feature selection is a greedy, iterative algorithm used to select the most relevant features for a predictive model. The objective of this algorithm is to improve the model's performance by identifying and using only the most relevant features, potentially reducing overfitting, improving accuracy, and reducing computational cost. 
    
    Implement forward feature selection using the following guidelines: 
    1. Start with an empty set of selected features.
    1. For each feature not yet in the selected set, do the following:
        1. Add the feature to the selected set temporarily.
        1. Train a model using the current set of selected features and evaluate its performance by calculating the cost or error on a validation set.
        1. Remove the temporarily added feature from the selected set.
    1. Choose the feature that resulted in the best model performance and permanently add it to the selected set.
    1. Repeat steps 2-3 until you have 5 features (not including the bias).
    '''
    number_of_features = X_train.shape[1]
    X_train = apply_bias_trick(X_train)
    X_val = apply_bias_trick(X_val)
    best_cost = np.inf
    best_feature = None

    for i in range(5):
        for feature in range(number_of_features):

            if feature in selected_features:
                continue

            selected_features.append(feature)
            theta = np.random.random(size=len(selected_features) + 1)  # adding 1 for bias
            x_train_to_use = X_train[:, [0] + selected_features]
            x_val_to_use = X_val[:, [0] + selected_features]
            gradient_descent_result = efficient_gradient_descent(x_train_to_use, y_train, theta, best_alpha, iterations)
            cost = compute_cost(x_val_to_use, y_val, gradient_descent_result[0])

            if cost < best_cost:
                best_cost = cost
                best_feature = feature

            selected_features.remove(feature)

        selected_features.append(best_feature)

    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly