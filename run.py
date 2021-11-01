import numpy as np

from RandomForest import RandomForestClassifier

# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score

from proj1_helpers import *
from utils import train_test_split


def feature_augmentation(X):

    return X


def remove_nan(X):
    mask = X == -999.
    mask = (1 - np.all(mask, axis=0)).astype(np.bool)
    return X[:, mask]


class StandardScaler(object):
    def __init__(self):
        self.std = None
        self.avg = None

    def fit(self, X):
        self.avg = X.mean(axis=0)
        self.std = X.std(axis=0)

    def transform(self, X):
        X = (X - self.avg) / self.std
        return X


class HiggsDataset():
    def __init__(self, dataset_dir):
        y, X, ids = load_csv_data(dataset_dir)

        y, X, ids = y[0:100], X[0:100], ids[0:100]

        y = (y == 1).astype(int)

        self.PRI_jet_nums = np.unique(X[:, 22])
        self.PRI_jet_nums.sort()

        self.subsets = []

        for i in self.PRI_jet_nums:
            X_i = X[(X[:, 22] == i).nonzero()]
            y_i = y[(X[:, 22] == i).nonzero()]
            X_i = remove_nan(X_i)
            X_i = feature_augmentation(X_i)
            ids_i = ids[(X[:, 22] == i).nonzero()]
            self.subsets.append((X_i, y_i, ids_i))


    def split(self, ratio=0.7):
        train_test_sets = []
        for i in self.subsets:
            train_test_sets.append(train_test_split(i[0], i[1], ratio=ratio, seed=1))
        return train_test_sets


class HiggsClassifier():
    def __init__(self):
        self.clfs = []
        self.scalers = []

    def train(self, dataset):
        train_test_sets = dataset.split()

        accuracy_all = 0
        test_num = 0

        for X_train, X_test, y_train, y_test in train_test_sets:

            print(X_train.shape)

            scaler = StandardScaler().fit(X_train)
            self.scalers.append(scaler)

            # X_train = scaler.transform(X_train)
            # X_test = scaler.transform(X_test)

            paras = {'n_estimators': 400, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 60, 'bootstrap': False}

            clf = RandomForestClassifier(**paras)

            clf.fit(X_train, y_train)

            # ------------
            # target_names = ['class b', 'class s']

            # pred = clf.predict(X_train)
            # print(classification_report(y_train, pred, target_names=target_names))
            # print(accuracy_score(y_train, pred))

            # # ------------

            # pred = clf.predict(X_test)
            # print(classification_report(y_test, pred, target_names=target_names))

            # accuracy = accuracy_score(y_test, pred)
            # print(accuracy)

            # accuracy_all += accuracy * len(y_test)
            # test_num += len(y_test)

            # # ------------
            self.clfs.append(clf)

        print('Overall acurracy:', accuracy_all / test_num)

    def predict(self, dataset):
        results = []
        for (X, _, ids), scaler, clf in zip(dataset.subsets, self.scalers, self.clfs):
            # X = scaler.transform(X)
            pred = clf.predict(X)
            pred = pred * 2 - 1
            results.append((ids, pred))

        print([i[0].squeeze() for i in results])

        preds = np.concatenate([i[1].squeeze() for i in results])
        ids = np.concatenate([i[0].squeeze() for i in results])


        idx = np.argsort(ids)

        preds = preds[idx]
        ids = ids[idx]

        create_csv_submission(ids, preds, 'sss.csv')




if __name__ == '__main__':
    train_dataset = HiggsDataset('project_1/train.csv')

    test_dataset = HiggsDataset('project_1/test.csv')

    clf = HiggsClassifier()

    clf.train(train_dataset)
    clf.predict(test_dataset)

