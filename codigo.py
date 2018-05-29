#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

#Constant
SEED = 8957
TRAIN_TEST_SPLIT = 0.3
CV_SPLITS = 5

def HotEncoder(feature):
    le = LabelEncoder()
    enc = OneHotEncoder(sparse=False)
    localFeature = feature.copy()
    localFeature = le.fit_transform(localFeature)
    localFeature = localFeature.reshape(len(localFeature), 1)
    localFeature = enc.fit_transform(localFeature)

    return localFeature


def Preprocessing(dataset):
    # Encode categorical features and cast integer to float 
    age_feature = dataset['age'].astype(np.float64).reshape([-1,1])
    job_feature = HotEncoder(dataset['job'])
    marital_feature = HotEncoder(dataset['marital'])
    education_feature = HotEncoder(dataset['education'])
    default_feature = HotEncoder(dataset['default'])
    balance_feature = dataset['balance'].astype(np.float64).reshape([-1,1])
    housing_feature = HotEncoder(dataset['housing'])
    loan_feature = HotEncoder(dataset['loan'])
    contact_feature = HotEncoder(dataset['contact'])
    day_feature = dataset['day'].astype(np.float64).reshape([-1,1])
    month_feature = HotEncoder(dataset['month'])
    duration_feature = dataset['duration'].astype(np.float64).reshape([-1,1])
    campaign_feature = dataset['campaign'].astype(np.float64).reshape([-1,1])
    pdays_feature = dataset['pdays'].astype(np.float64).reshape([-1,1])
    previous_feature = dataset['previous'].astype(np.float64).reshape([-1,1])
    poutcome_feature = HotEncoder(dataset['poutcome'])

    # Build X dataset
    X = np.concatenate([
        # Scalar features
        age_feature,
        balance_feature,
        day_feature,
        duration_feature,
        campaign_feature,
        pdays_feature,
        previous_feature,

        # Catergorical features
        job_feature,
        marital_feature,
        education_feature,
        default_feature,
        housing_feature,
        loan_feature,
        contact_feature,
        month_feature,
        poutcome_feature
    ], axis=1)

    scalarLimitIndex = 6

    # Build y dataset:
    y = np.where(dataset['y'] == '"yes"', 1, -1)

    return X, y, scalarLimitIndex
 


dataset = np.genfromtxt('datos/bank-full.csv', delimiter=';', dtype=None, names=True, encoding='utf-8')
X, y, scalarLimitIndex = Preprocessing(dataset)

print("N: ", X.shape[0])
print("Number of features: ", X.shape[1])
print("Number of 'yes': ", len(y[y==1]))
print("Number of 'no': ", len(y[y==-1]))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=TRAIN_TEST_SPLIT, random_state=SEED)

X_train_scaled = X_train.copy()
X_train_scaled[:, :scalarLimitIndex] = StandardScaler().fit_transform(X_train[:,:scalarLimitIndex])

X_test_scaled = X_test.copy()
X_test_scaled[:, :scalarLimitIndex] = StandardScaler().fit_transform(X_test[:,:scalarLimitIndex])

# Regularization hyperparameter:
C = np.logspace(-4, 4, 10)
# kflod for CV
kfold = model_selection.KFold(n_splits=CV_SPLITS)

print("\Findig C hyperparameter for Logistic Regression...")

penalty = ['l2']
hyperparam = dict(C=C, penalty=penalty)

LR = linear_model.LogisticRegression()
clf = GridSearchCV(LR, hyperparam, cv=kfold, scoring='accuracy', n_jobs=-1)
model = clf.fit(X_train_scaled, y_train)

LR_C = model.best_estimator_.get_params()['C']
 
print("\tBest C for Logistic Regression: ",  LR_C)
print("\tScore: ", model.best_score_)

"""
print("\nFinding C hyperparameter for SVM...")

kernels= ['rbf', 'poly']
gamma = np.logspace(-2, 1, 1)
hyperparam = dict(C=C, kernel=kernels, gamma=gamma)

SVM = SVC(kernel='rbf')
clf = GridSearchCV(SVM, hyperparam, cv=kfold, scoring='accuracy', n_jobs=-1)
model = clf.fit(X_train_scaled, y_train)

SVM_C = model.best_estimator_.get_params()['C']
SVM_GAMMA = model.best_estimator_.get_params()['gamma']
SVM_KERNEL = model.best_estimator_.get_params()['kernel']

print("\tBest C for SVM: ",  SVM_C)
print("\tBest gamma for SVM: ",  SVM_GAMMA)
print("\tBest kernel for SVM: ",  SVM_KERNEL)
print("\tScore: ", model.best_score_)
"""

print("\nFinding optimal number of trees in Random Forest...")

NumberOfTrees = np.arange(5,11)

  








