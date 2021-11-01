"""
The implementation of random forest binary classifier. The code is based on:
https://github.com/zhaoxingfeng/RandomForest

Improvement and modifications are made, including:
    * improvement on implementation efficiency
    * removal of dependency on Pandas
    * modifying to scikit-learn APIs
    * add of entropy criterion
    * shift from python 2 to python 3
    * modifications for better clarity and simplification

The External library joblib is used for parallel computing. It can be safely
removed, at the cost of largely increasing training time.

We compare the implemented random forest classifier with
sklearn.ensemble.RandomForestClassifier. Given the same dataset and parameters,
our implementation can achieve approximate results. However, it takes much
longer time for training. In our test, when the RandomForestClassifier in
sklearn only takes few seconds for the training, our implementation takes ~2h
for the training.
"""

import pandas as pd
import numpy as np
import random
import math
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split


class Tree(object):
    """ Implementation of decision trees."""
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.tree_left = None
        self.tree_right = None

    def calc_predict_value(self, X):
        if self.leaf_value is not None:
            return self.leaf_value
        elif X[self.split_feature] <= self.split_value:
            return self.tree_left.calc_predict_value(X)
        else:
            return self.tree_right.calc_predict_value(X)


class RandomForestClassifier(object):
    """ Numpy implementation of random forest binary classifier.


    """
    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 criterion='gini',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_split_gain=0.0,
                 max_features='auto',
                 max_samples=0.8,
                 random_state=None,
                 n_jobs=-1):
        """ Initilization of the random forest classifier.

        Use the sklearn-like API.

        Paras:
            n_estimators:       Number of decision trees. Default is 100.
            max_depth:          The maximum depth of decision trees.
            min_samples_split:  Minimum samples needed for the splitting.
            min_samples_leaf:   Minimum samples needed for leaves.
            min_split_gain:     Minimum gain needed for further splitting
            max_features:       Maximum number of features in feature sampling
            max_samples:        Maximum number of samples in subsampling
            random_state:       Random state of seed.
            n_jobs:             Number of threads to use.
        """
        self.n_estimators = n_estimators
        self.max_depth = float('inf') if max_depth is None else max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_split_gain = min_split_gain
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = 1 if n_jobs is None else n_jobs

        self.trees = None
        self.feature_importances_ = dict()

    def fit(self, X, y):
        """ Train the random forest classifier.
        
        Paras:
            X:  Feature matrix. The shape is [n_samples, n_features]
            y:  Labels. The shape is [n_samples, ], and the value can only be
                either 0 or 1.
        """

        feature_num = X.shape[1]

        assert len(np.unique(y)) == 2

        if self.random_state is True:
            random.seed(self.random_state)
        random_state_stages = random.sample(range(self.n_estimators), self.n_estimators)

        if self.max_features == "auto":
            self.max_features = int(feature_num ** 0.5)
        elif self.max_features == "sqrt":
            self.max_features = int(feature_num ** 0.5)
        elif self.max_features == "log2":
            self.max_features = int(math.log(feature_num))
        else:
            self.max_features = feature_num

        self.trees = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(self._parallel_build_trees)(X, y, random_state)
                for random_state in random_state_stages
        )
        
    def _parallel_build_trees(self, X, y, random_state):
        """ Build decision trees in parallel.
        """
        sample_num = X.shape[0]
        indexes = np.arange(0, sample_num)
        np.random.seed(random_state)
        np.random.shuffle(indexes)
        indexes = indexes[0:int(self.max_samples*sample_num)]

        tree = self._build_single_tree(X[indexes, :], y[indexes], depth=0)
        return tree

    def _build_single_tree(self, X, y, depth):
        """ Build decision tree recursively.
        """
        if len(np.unique(y)) <= 1 or X.shape[0] <= self.min_samples_split:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(y)
            return tree

        if depth < self.max_depth:
            best_split_feature, best_split_value, best_split_gain = self.choose_best_feature(X, y)
            left_dataset, right_dataset, left_targets, right_targets = \
                self.split_dataset(X, y, best_split_feature, best_split_value)

            tree = Tree()

            if left_dataset.__len__() <= self.min_samples_leaf or \
                    right_dataset.__len__() <= self.min_samples_leaf or \
                    best_split_gain <= self.min_split_gain:
                tree.leaf_value = self.calc_leaf_value(y)
                return tree
            else:
                self.feature_importances_[best_split_feature] = \
                    self.feature_importances_.get(best_split_feature, 0) + 1

                tree.split_feature = best_split_feature
                tree.split_value = best_split_value
                tree.tree_left = self._build_single_tree(left_dataset, left_targets, depth + 1)
                tree.tree_right = self._build_single_tree(right_dataset, right_targets, depth + 1)
                return tree
        else:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(y)
            return tree

    def choose_best_feature(self, X, y):
        """ Choose best features based on Gini or entropy.
        """
        best_split_gain = 1
        best_split_feature = None
        best_split_value = None

        for i in range(X.shape[1]):
            unique_values = sorted(list(np.unique(X[:, i])))
            if len(np.unique(X[:, i])) > 100:
                unique_values = [unique_values[int(j)] for j in np.linspace(0, len(unique_values) - 1, 100)]

            for split_value in unique_values:

                left_targets = y[list((X[:, i] <= split_value).nonzero()[0])]
                right_targets = y[list((X[:, i] > split_value).nonzero()[0])]

                split_gain = self.calc_split_point(left_targets, right_targets, self.criterion)

                if split_gain < best_split_gain:
                    best_split_feature = i
                    best_split_value = split_value
                    best_split_gain = split_gain

        return best_split_feature, best_split_value, best_split_gain


    @staticmethod
    def calc_leaf_value(y):
        if y.sum() / len(y) < 0.5:
            return 0
        else:
            return 1

    @staticmethod
    def calc_split_point(left_targets, right_targets, method):
        split_gain = 0
        for targets in [left_targets, right_targets]:
            pos_prob = targets.sum() / len(targets) if len(targets) != 0 else 0.5
            neg_prob = 1 - pos_prob

            if method == 'gini':   
                gini = 1 - pos_prob ** 2 - neg_prob ** 2
                split_gain += len(targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini
            elif method == 'entropy':
                split_gain += - pos_prob * np.log(pos_prob) - neg_prob * np.log(neg_prob)

        return split_gain

    @staticmethod
    def split_dataset(X, y, split_feature, split_value):
        left_indexes = list((X[:, split_feature] <= split_value).nonzero()[0])
        right_indexes = list((X[:, split_feature] > split_value).nonzero()[0])
        left_dataset = X[left_indexes, :]
        left_targets = y[left_indexes]
        right_dataset = X[right_indexes, :]
        right_targets = y[right_indexes]
        return left_dataset, right_dataset, left_targets, right_targets

    def predict(self, X):
        """ Predict given the X.
        """
        preds = []
        for row in X:
            pred_list = []
            for tree in self.trees:
                pred_list.append(tree.calc_predict_value(row))

            preds.append(1 if np.sum(pred_list) / len(pred_list) > 0.5 else 0)

        preds = np.array(preds)

        return preds

import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    df = pd.read_csv("project_1/train.csv")

    df = df.drop(columns=['PRI_jet_all_pt'])
    df = df.drop(columns=['PRI_tau_phi'])
    df = df.drop(columns=['PRI_lep_phi'])
    df = df.drop(columns=['PRI_met_phi'])
    df = df.drop(columns=['PRI_jet_leading_phi'])
    df = df.drop(columns=['PRI_jet_subleading_phi'])

    X =  df.to_numpy()[:, 2:].astype(np.float64)
    y =  (df.to_numpy()[:, 1] == 'b').astype(np.float64)
    print(X.shape, y.shape)

    clf = RandomForestClassifier(n_estimators=16, random_state=66)

    # clf = RandomForestClassifier(n_estimators=100,
    #                              max_depth=5,
    #                              min_samples_split=6,
    #                              min_samples_leaf=2,
    #                              min_split_gain=0.0,
    #                              max_features="sqrt",
    #                              max_samples=0.8,
    #                              random_state=66)

    train_features, test_features, train_labels, test_labels = train_test_split(X, y, train_size=0.7)

    clf.fit(train_features, train_labels)

    joblib.dump(clf, 'aaa')

    preds = clf.predict(test_features)

    print(classification_report(y_true=test_labels, y_pred=preds))
    print(accuracy_score(y_true=test_labels, y_pred=preds))

    # clf.fit(pd.DataFrame(train_features), pd.Series(train_labels.squeeze()))


