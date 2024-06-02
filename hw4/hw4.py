import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = 0.0
    mu_x = np.mean(x)
    mu_y = np.mean(y)

    mone = np.sum((x - mu_x) * (y - mu_y))
    mechane_sums = np.sum((x - mu_x) ** 2) * np.sum((y - mu_y) ** 2)
    mechane = np.sqrt(mechane_sums)

    r = mone / mechane
    return r

def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    best_features = []
    X["date"] = pd.to_numeric(pd.to_datetime(X["date"]))
    X["id"] = pd.to_numeric(pd.to_datetime(X["id"]))

    correlations_by_feature = {}

    amount_of_features = X.shape[1]
    for i in range(amount_of_features):
        feature_name = X.columns[i]
        pearson_correlation_for_feature = pearson_correlation(X.iloc[:, i], y)
        correlations_by_feature[feature_name] = pearson_correlation_for_feature

    sorted_correlations_by_feature = sorted(correlations_by_feature.items(), key=lambda x: abs(x[1]), reverse=True)
    best_features = [x[0] for x in sorted_correlations_by_feature[:n_features]]

    return best_features

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        np.random.seed(self.random_state)
        X = np.column_stack((np.ones(X.shape[0]), X))  # Add bias trick
        m, n = X.shape
        self.theta = np.random.rand(n)
        for iteration in range(self.n_iter):
            h = self.sigmoid_function(X)
            J = self.cost_function(h, m, y)
            self.Js.append(J)
            gradient = self.calculate_gradient(X, h, m, y)
            self.theta = self.calculate_theta(gradient)
            self.thetas.append(self.theta.copy())
            if self.has_no_impact():
                break

    def cost_function(self, h, m, y):
        return 1 / m * (-y @ np.log(h) - (1 - y) @ np.log(1 - h))

    def calculate_theta(self, gradient):
        return self.theta - (self.eta * gradient)

    def calculate_gradient(self, X, h, m, y):
        return (1 / m) * X.T @ (h - y)

    def has_no_impact(self):
        ret_val = False

        if len(self.Js) > 1:
            ret_val = np.abs(self.Js[-1] - self.Js[-2]) < self.eps

        return ret_val

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        X = np.column_stack((np.ones(X.shape[0]), X))  # Add bias trick
        h = self.sigmoid_function(X)
        preds = (h >= 0.5).astype(int)
        return preds

    def sigmoid_function(self, X):
        return 1 / (1 + np.exp(-(X @ self.theta.T)))


def calculate_accuracy(y_true, y_prediction):
    """
    Calculate the accuracy of the predictions.
    """
    correct_predictions = np.sum(y_true == y_prediction)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions

    return accuracy


def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None
    np.random.seed(random_state)
    dataset = np.column_stack((X, y))
    np.random.shuffle(dataset)
    subsets = np.array_split(dataset, folds)
    accuracies = []

    # For each subset, treat one subset in size of 100/folds% as the test set and
    # the other subsets will be the training set
    for i in range(folds):
        current_validation_set = subsets[i]
        # Ensure the train set fold will not include the current fold test set
        current_training_set = np.vstack([subset for j, subset in enumerate(subsets) if j != i])

        X_train, y_train = current_training_set[:, :-1], current_training_set[:, -1]
        X_val, y_val = current_validation_set[:, :-1], current_validation_set[:, -1]

        algo.fit(X_train, y_train)
        y_prediction = algo.predict(X_val)

        accuracy = calculate_accuracy(y_val, y_prediction)
        accuracies.append(accuracy)

    cv_accuracy = np.mean(accuracies)
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(-0.5 * ((data - mu) / sigma) ** 2)
    p = coefficient * exponent

    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """ Initialize parameters randomly based on the data distribution """
        print(data)
        print(data[0])
        self.weights = np.ones(self.k) / self.k  # Start with equal weights
        self.mus = np.random.choice(data, self.k, replace=False)  # Randomly chosen means from the data
        self.sigmas = np.std(data) * np.random.rand(self.k) + 1  # Random standard deviations

    def expectation(self, data):
        """ E-step: compute responsibilities, i.e., the probabilities of each component given the data """
        n = data.shape[0]
        self.responsibilities = np.zeros((n, self.k))
        for j in range(self.k):
            self.responsibilities[:, j] = self.weights[j] * norm_pdf(data, self.mus[j], self.sigmas[j])
        self.responsibilities /= self.responsibilities.sum(axis=1, keepdims=True)

    def maximization(self, data):
        """ M-step: update the parameters of the Gaussians """
        weights = self.responsibilities.sum(axis=0)
        self.weights = weights / data.size
        self.mus = np.sum(data[:, np.newaxis] * self.responsibilities, axis=0) / weights
        self.sigmas = np.sqrt(np.sum(self.responsibilities * (data[:, np.newaxis] - self.mus) ** 2, axis=0) / weights)

    def fit(self, data):
        """ Fit the model to the data using the EM algorithm """
        data = data[:,0]
        self.init_params(data)
        log_likelihood_old = 0
        for iteration in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)
            log_likelihood_new = np.sum(np.log(np.sum(self.weights * norm_pdf(data[:, np.newaxis], self.mus, self.sigmas), axis=1)))
            if np.abs(log_likelihood_new - log_likelihood_old) < self.eps:
                break
            log_likelihood_old = log_likelihood_new

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf_is = [normal_pdf(data, mus[i], sigmas[i]) for i in range(len(mus))]
    return sum(pdf_is[i] * weights [i] for i in range(len(mus)))

def normal_pdf(x, mu, sigma):
    denominator = sigma * ((2*np.pi)**0.5)
    numerator  = np.exp((-(x-mu)**2) / 2*(sigma**2))
    return numerator/ denominator

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None
    # Todo maor - Provide one or two sentences on each graph explaining what you observe in the graph.
    # Todo maor - uncomment Naive Bayes code
    lor = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor.fit(x_train, y_train)

    # nb = NaiveBayesGaussian(k=k)
    # nb.fit(x_train, y_train)

    lor_train_acc = calculate_accuracy(y_train, lor.predict(x_train))
    lor_test_acc = calculate_accuracy(y_test, lor.predict(x_test))
    # bayes_train_acc = calculate_accuracy(y_train, nb.predict(x_train))
    # bayes_test_acc = calculate_accuracy(y_test, nb.predict(x_test))

    plot_decision_regions(x_train, y_train, lor, title="Logistic Regression Decision Boundary")
    # plot_decision_regions(x_train, y_train, nb, title="Naive Bayes Decision Boundary")

    # Plot cost vs iteration for Logistic Regression
    plt.plot(lor.Js)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Logistic Regression - Cost vs Iteration')
    plt.show()

    print('Logistic Regression - Train Accuracy: ', lor_train_acc)
    print('Logistic Regression - Test Accuracy: ', lor_test_acc)
    # print('Naive Bayes - Train Accuracy: ', bayes_train_acc)
    # print('Naive Bayes - Test Accuracy: ', bayes_test_acc)

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }


# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):
    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.show()