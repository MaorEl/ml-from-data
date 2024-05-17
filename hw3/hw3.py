import numpy as np
import math
###### Your ID ######
# ID1: 312412927
# ID2: 204226815
#####################

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.1,
            (0, 1): 0.25,
            (1, 0): 0.25,
            (1, 1): 0.4
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.045,
            (0, 0, 1): 0.045,
            (0, 1, 0): 0.105,
            (0, 1, 1): 0.105,
            (1, 0, 0): 0.105,
            (1, 0, 1): 0.105,
            (1, 1, 0): 0.245,
            (1, 1, 1): 0.245,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        for x in self.X:
            for y in self.Y:
                p_x_given_y = X_Y[(x, y)] / Y[y]
                if not np.isclose(X[x] * Y[y], p_x_given_y):
                    return True
        return False

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        for c in C:
            for x in X:
                p_x_given_c = X_C[(x, c)] / C[c]
                for y in Y:
                    p_y_given_c = Y_C[(y, c)] / C[c]
                    p_x_y_given_c = X_Y_C[(x, y, c)] / C[c]
                    if (not np.isclose(p_x_given_c * p_y_given_c, p_x_y_given_c)):
                        print(
                            f"for x:{x}, y:{y}. p_x_y_given_c={c}: {p_x_y_given_c} while p_x_given_c={c}:{p_x_given_c}, p_y_given_c={c}:{p_y_given_c}")
                        return False
        return True

    def _are_all_probabilities_valid(self):
        if not np.isclose(sum(self.X.values()), 1):
            return False
        if not np.isclose(sum(self.Y.values()), 1):
            return False
        if not np.isclose(sum(self.C.values()), 1):
            return False
        if not np.isclose(sum(self.X_Y.values()), 1):
            return False
        if not np.isclose(sum(self.X_C.values()), 1):
            return False
        if not np.isclose(sum(self.Y_C.values()), 1):
            return False
        if not np.isclose(sum(self.X_Y_C.values()), 1):
            return False
        return True


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = k * math.log(rate) - rate - math.lgamma(k + 1)
    return log_p


def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = []
    for rate in rates:
        log_likelihood = np.sum([poisson_log_pmf(k, rate) for k in samples])
        likelihoods.append(log_likelihood)

    return likelihoods


def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates)  # might help
    max_likelihood_index = np.argmax(likelihoods)
    rate = rates[max_likelihood_index]

    return rate


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = np.mean(samples)
    return mean


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = (1 / (np.sqrt(2 * np.pi * std ** 2))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

    return p


class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        self.amount_of_instances = len(dataset)
        self._class_data = dataset[dataset[:, -1] == class_value, :-1]
        self._mean = np.mean(self._class_data, axis=0)
        self._std = np.std(self._class_data, axis=0)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        class_instances = len(self._class_data)
        prior = class_instances / self.amount_of_instances

        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = 1
        for i in range(len(x)):
            likelihood *= normal_pdf(x[i], self._mean[i], self._std[i])
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior() * self.get_instance_likelihood(x)
        return posterior


class MAPClassifier():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self._ccd0 = ccd0
        self._ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        posterior_class_0 = self._ccd0.get_instance_posterior(x)
        posterior_class_1 = self._ccd1.get_instance_posterior(x)

        if posterior_class_0 > posterior_class_1:
            ret_val = 0
        else:
            ret_val = 1

        return ret_val


def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    correct_predictions = 0
    for instance in test_set:
        x = instance[:-1]
        actual_class = instance[-1]
        predicted_class = map_classifier.predict(x)
        if predicted_class == actual_class:
            correct_predictions += 1

    test_set_size = len(test_set)
    acc = correct_predictions / test_set_size

    return acc


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    d = len(x)
    x_minus_mean = x - mean
    cov_inv = np.linalg.inv(cov)
    exp_term = np.exp(-0.5 * np.dot(x_minus_mean.T, np.dot(cov_inv, x_minus_mean)))
    sqrt = np.sqrt(np.linalg.det(cov))
    power = np.power(2 * np.pi, d / 2)

    pdf = (1 / (power * sqrt)) * exp_term

    return pdf


class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self._all_instances = len(dataset)
        self._class_data = dataset[dataset[:, -1] == class_value, :-1]
        self._mean = np.mean(self._class_data, axis=0)
        self._covariance = np.cov(self._class_data, rowvar=False)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        class_instances = len(self._class_data)
        prior = class_instances / self._all_instances

        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = multi_normal_pdf(x, self._mean, self._covariance)
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior() * self.get_instance_likelihood(x)

        return posterior


class MaxPrior():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self._ccd0 = ccd0
        self._ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        prior_class_0 = self._ccd0.get_prior()
        prior_class_1 = self._ccd1.get_prior()

        if prior_class_0 > prior_class_1:
            ret_val = 0
        else:
            ret_val = 1

        return ret_val


class MaxLikelihood():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self._ccd0 = ccd0
        self._ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        likelihood_class_0 = self._ccd0.get_instance_likelihood(x)
        likelihood_class_1 = self._ccd1.get_instance_likelihood(x)

        if likelihood_class_0 > likelihood_class_1:
            ret_val = 0
        else:
            ret_val = 1

        return ret_val


EPSILLON = 1e-6  # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.


class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self._all_instances = len(dataset)
        self._class_data = dataset[dataset[:, -1] == class_value, :-1]
        self._num_instances = self._class_data.shape[0]
        self._num_features = self._class_data.shape[1]
        dataset_without_class_column = dataset[:-1]  # removing the class column
        self._unique_feature_values = [np.unique(dataset_without_class_column[:, i]) for i in range(self._num_features)]
        self._feature_counters = [np.bincount(self._class_data[:, i].astype(int),
                                              minlength=len(self._unique_feature_values[i]))
                                  for i in range(self._num_features)]

    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        class_instances = len(self._class_data)
        prior = class_instances / self._all_instances

        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = 1

        for i in range(self._num_features):
            feature_index = np.where(self._unique_feature_values[i] == x[i])[0]
            if feature_index.size == 0:
                likelihood *= EPSILLON
            else:
                feature_count = self._feature_counters[i][feature_index[0]]
                likelihood *= (feature_count + 1) / (self._num_instances + len(self._unique_feature_values[i]))

        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior() * self.get_instance_likelihood(x)

        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self._ccd0 = ccd0
        self._ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """

        posterior_class_0 = self._ccd0.get_instance_posterior(x)
        posterior_class_1 = self._ccd1.get_instance_posterior(x)

        if posterior_class_0 > posterior_class_1:
            ret_val = 0
        else:
            ret_val = 1

        return ret_val

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        correct_predictions = 0
        for instance in test_set:
            x = instance[:-1]
            actual_class = instance[-1]
            predicted_class = self.predict(x)

            if predicted_class == actual_class:
                correct_predictions += 1

        test_set_size = len(test_set)
        accuracy = correct_predictions / test_set_size

        return accuracy
