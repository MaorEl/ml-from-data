import numpy as np
import matplotlib.pyplot as plt
import math

###### Your ID ######
# ID1: 312412927
# ID2: 204226815
#####################

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}


def calc_gini(data):
    labels_to_count = get_label_frequencies(data)
    total = sum(labels_to_count.values())
    return 1 - sum((count / total) ** 2 for count in labels_to_count.values())


def calc_entropy(data):
    labels_to_count = get_label_frequencies(data)
    total = sum(labels_to_count.values())
    return -1 * sum((count / total) * math.log2(count / total) for count in labels_to_count.values())


def get_label_frequencies(data):
    labels_to_count = {}
    labels_values = data[:, -1]
    for value in labels_values:
        labels_to_count[value] = labels_to_count.get(value, 0) + 1
    return labels_to_count


class DecisionNode:
    def __init__(
            self,
            data,
            impurity_func,
            feature=-1,
            depth=0,
            chi=1,
            max_depth=1000,
            gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        
        label_freqs = get_label_frequencies(self.data)
        if not label_freqs:
            return None
        return max(label_freqs, key=label_freqs.get)
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        if self.feature < 0 or self.feature >= self.data.shape[1] - 1 or self.terminal:
            self.feature_importance = 0
            return

        node_goodness, _ = self.goodness_of_split(self.feature)
        node_probe = (len(self.data) / n_total_sample)
        self.feature_importance = node_probe * node_goodness

    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        if self.terminal:
            return 0, {}
                
        # create data subsets per feature_value
        groups = {}  # groups[feature_value] = data_subset
        feature_values = self.data[:, feature]
        for index in range(len(feature_values)):
            value = feature_values[index]
            new_row = self.data[index]
            if value not in groups:
                groups[value] = []
            groups[value].append(new_row)

        # convert to numpy matrix
        for feature_values in groups:
            groups[feature_values] = np.array(groups[feature_values])

        # calculate goodness of split
        total = len(self.data)
        if total == 0:
            return 0, {}

        # Calculate initial impurity
        initial_impurity = self.impurity_func(self.data)
        weighted_impurity = 0
        split_info = 0

        for group in groups.values():
            proportion = len(group) / total
            weighted_impurity += proportion * self.impurity_func(group)
            if proportion > 0:
                split_info += -proportion * np.log2(proportion)

        goodness = initial_impurity - weighted_impurity

        if not self.gain_ratio or split_info == 0:
            return goodness, groups

        gain_ratio = goodness / split_info
        return gain_ratio, groups

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """

        if self.max_depth == self.depth or self.terminal:
            self.terminal = True
            return

        max_goodness, groups_of_max_goodness, max_goodness_feature_index = 0, {}, -1
        for feature_index in range(len(self.data[0]) - 1):
            goodness, groups = self.goodness_of_split(feature_index)
            if goodness > max_goodness:
                max_goodness, groups_of_max_goodness, max_goodness_feature_index = goodness, groups, feature_index

        if max_goodness == 0 or not groups_of_max_goodness:
            self.terminal = True
            return

        self.feature = max_goodness_feature_index
        number_of_feature_values = len(groups_of_max_goodness)
        number_of_labels = len(np.unique(self.data[:, -1]))
        df = (number_of_feature_values - 1) * (number_of_labels - 1)
        print(
            f'in node in depth: {self.depth} with feature: {self.feature} getting chi score for df: {df}. '
            f'number_of_features:{number_of_feature_values}, number_of_labels: {number_of_labels}')

        if self.chi != 1:
            significance_level = self.chi
            chi_square = self.get_node_chi_square(groups_of_max_goodness)
            z_score = chi_table.get(df, {}).get(significance_level, float('inf'))

            if chi_square < z_score:
                print(
                    f"chi square value is not !!!!!!! significant in p-val = {self.chi}. "
                    f"chi square: {chi_square} while threshold value: {z_score}. Pruning.")
                self.terminal = True
                return
            else:
                print(
                    f"chi square value is significant in p-val = {self.chi}. chi square: {chi_square} "
                    f"while threshold value: {z_score}. Not Pruning.")

        else:
            print(f"chi value is 1. Not Pruning.")

        for value, group in groups_of_max_goodness.items():
            group_node = DecisionNode(data=group, feature=max_goodness_feature_index, impurity_func=self.impurity_func,
                                      gain_ratio=self.gain_ratio, depth=self.depth + 1, max_depth=self.max_depth,
                                      chi=self.chi)
            print(f"Node in depth: {self.depth} creating child according to feature: {self.feature} and value: {value}")
            self.add_child(group_node, value)

        print("\n\n")

    def get_node_chi_square(self, groups_of_max_goodness):
        labels_count = get_label_frequencies(self.data)
        labels_prob = {label_name: label_count / len(self.data) for label_name, label_count in labels_count.items()}
        # [P(Y=0), P(Y=1) ... P(Y=k-1)] where k - number of feature values
        chi_square = 0 
        for group in groups_of_max_goodness.values():  # each group is a subset of max goodness feature
            d_f = len(group)
            feature_labels_count = get_label_frequencies(group) # pf:cnt0, nf:cnt1 ,,,, 
            for key in labels_prob.keys():
                if feature_labels_count.get(key) is None:
                    feature_labels_count[key] = 0
            expected_labels = {label_name: d_f * label_prob for label_name, label_prob in labels_prob.items()}
            #  [d_f*P(Y=0), d_f*P(Y=1) ... d_f*P(Y=k-1)]
            chi_square += sum(
                ((feature_labels_count[label_name] - expected_labels[label_name]) ** 2) / expected_labels[label_name]
                for label_name in expected_labels.keys())
        return chi_square


class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = DecisionNode(data=self.data, impurity_func=self.impurity_func, gain_ratio=self.gain_ratio, depth=0,
                                 max_depth=self.max_depth, chi=self.chi)
        self.build_tree_recursive(self.root, len(self.data))
        print(f'Finished building tree')

    def build_tree_recursive(self, decision_node: DecisionNode, data_size):
        print(f'Building node in depth: {decision_node.depth} and feature: {decision_node.feature}')
        decision_node.split()
        decision_node.calc_feature_importance(data_size)
        for child_node in decision_node.children:
            self.build_tree_recursive(child_node, data_size)

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        node: DecisionNode = self.root

        while not node.terminal:
            feature_value = instance[node.feature]
            try:
                index = node.children_values.index(feature_value)
                node = node.children[index]
            except ValueError:
                break

        pred = node.pred
        return pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        correct = 0
        for instance in dataset:
            if self.predict(instance) == instance[-1]:
                correct += 1

        accuracy = correct / len(dataset)
        return accuracy
        
    def depth(self):
        return self.root.depth()

    def calculate_tree_depth(self, node=None):
        if node is None:
            node = self.root

        if node.terminal:
            return node.depth

        return max(self.calculate_tree_depth(child) for child in node.children)


def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree_entropy_gain_ratio = DecisionTree(data=X_train, impurity_func=calc_entropy,
                                               gain_ratio=True, max_depth=max_depth)
        tree_entropy_gain_ratio.build_tree()
        train_accuracy = tree_entropy_gain_ratio.calc_accuracy(X_train)
        training.append(train_accuracy)
        test_accuracy = tree_entropy_gain_ratio.calc_accuracy(X_validation)
        validation.append(test_accuracy)
        print(f"Accuracies with max depth: {max_depth}. Train accuracy: {train_accuracy}, "
              f"Validation accuracy: {test_accuracy}")

    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc = []
    depth = []

    p_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]

    for p_value in p_values:
        tree_entropy_gain_ratio = DecisionTree(data=X_train, impurity_func=calc_entropy,
                                               gain_ratio=True, chi=p_value)
        tree_entropy_gain_ratio.build_tree()
        train_accuracy = tree_entropy_gain_ratio.calc_accuracy(X_train)
        chi_training_acc.append(train_accuracy)
        test_accuracy = tree_entropy_gain_ratio.calc_accuracy(X_test)
        chi_validation_acc.append(test_accuracy)
        tree_depth = tree_entropy_gain_ratio.calculate_tree_depth()
        depth.append(tree_depth)
        print(f"Accuracies with p-value: {p_value}. Depth: {tree_depth} Train accuracy: {train_accuracy}, "
              f"Test accuracy: {test_accuracy}")

    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    number_of_nodes = 0

    if node is not None:
        number_of_nodes += 1
        for child in node.children:
            number_of_nodes += count_nodes(child)

    return number_of_nodes






