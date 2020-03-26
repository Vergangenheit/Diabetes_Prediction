import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import numpy as np


def x_y(data: pd.DataFrame):
    y = data.response.values
    X = data.drop('response', axis=1).values

    return X, y


def fit_test_logmodel(X, y):
    logit_model = sm.Logit(y, X)
    result = logit_model.fit()
    print(result.summary())


def split_tt(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
    logreg = LogisticRegression(random_state=0)
    logreg.fit(X_train, y_train)

    return logreg


def score(logreg, X_test: np.array, y_test: np.array):
    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    return y_pred


def conf_matrix(y_test: np.array, y_pred: np.array):
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(confusion_matrix)


def class_report(y_test: np.array, y_pred: np.array):
    print(metrics.classification_report(y_test, y_pred))
