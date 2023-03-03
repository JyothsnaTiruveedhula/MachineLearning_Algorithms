'''from collections import Counter

import numpy as np

from decisionTree import DecisionTree


def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    print(X)
    return X[idxs], y[idxs]


def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=self.n_feats,
            )
            print(y)
            X_samp, y_samp = bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)


# Testing
if __name__ == "__main__":
    # Imports
    import pandas as pd
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    dataframe = pd.read_csv("Lab5\\drug.csv")
    X = dataframe.drop('Drug', axis='columns')
    y = dataframe.drop(['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'], axis= 'columns')

    from sklearn.preprocessing import LabelEncoder

    le_sex = LabelEncoder()
    le_bp = LabelEncoder()
    le_chol = LabelEncoder()
    le_drug = LabelEncoder()

    X['Sex_n'] = le_sex.fit_transform(X['Sex'])
    X['BP_n'] = le_bp.fit_transform(X['BP'])
    X['Cholestrol_n'] = le_chol.fit_transform(X['Cholesterol'])
    y['Drug_n'] = le_drug.fit_transform(y['Drug'])

    X_n = X.drop(['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'], axis= 'columns')
    y_n = y.drop('Drug', axis= 'columns')

    X_train, X_test, y_train, y_test = train_test_split(X_n, y_n, test_size=0.3, random_state=1234)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = RandomForest(n_trees=3, max_depth=10)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print("Accuracy:", acc)'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv("Lab5\\drug.csv")
X = dataframe.drop('Drug', axis='columns')
y = dataframe.drop(['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'], axis= 'columns')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_chol = LabelEncoder()
le_drug = LabelEncoder()

X['Sex_n'] = le_sex.fit_transform(X['Sex'])
X['BP_n'] = le_bp.fit_transform(X['BP'])
X['Cholestrol_n'] = le_chol.fit_transform(X['Cholesterol'])
y['Drug_n'] = le_drug.fit_transform(y['Drug'])

X_n = X.drop(['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'], axis= 'columns')
y_n = y.drop('Drug', axis= 'columns')

X_train, X_test, y_train, y_test = train_test_split(X_n, y_n, test_size=0.3)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=5)

regressor.fit(X_train, y_train)

regressor.predict(X_test)

accuracy = regressor.score(X_test, y_test)

print("\nAccuracy for Drug dataset is: ", accuracy)

dataframe = pd.read_csv("Lab5\\adult.csv")
X = dataframe.drop('US', axis='columns')
y = dataframe.drop(['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'ShelveLoc', 'Age', 'Education', 'Urban'], axis= 'columns')

le_shelve = LabelEncoder()
le_urban = LabelEncoder()
le_us = LabelEncoder()

X['ShelveLoc_n'] = le_shelve.fit_transform(X['ShelveLoc'])
X['Urban_n'] = le_urban.fit_transform(X['Urban'])
y['US_n'] = le_us.fit_transform(y['US'])

X_n = X.drop(['ShelveLoc', 'Urban'], axis= 'columns')
y_n = y.drop('US', axis= 'columns')

X_train, X_test, y_train, y_test = train_test_split(X_n, y_n, test_size=0.3)

regressor = RandomForestRegressor(n_estimators=5)

regressor.fit(X_train, y_train)

regressor.predict(X_test)

accuracy = regressor.score(X_test, y_test)

print("\n\nAccuracy for Adult dataset is: ", accuracy)