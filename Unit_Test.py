import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv("data/data_withfeatures.csv")
X = df.drop('Status', axis=1)
y = df['Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
params = {
    'n_estimators': [300, 350, 400, 450, 500, 550, 600, 650, 700],
    'learning_rate': [0.01, 0.05, 0.10, 0.15, 0.20],
    'max_depth': [5, 6, 7, 8]
}
import time
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import f1_score
xgb_classifier = XGBClassifier()
grid = GridSearchCV(estimator=xgb_classifier, param_grid=params, scoring='roc_auc', cv=10, n_jobs=-1,
                    verbose=3)
grid.fit(X_train, y_train)
print('\n Best estimator:')
print(grid.best_estimator_)
print('\n Best score:')
print(grid.best_score_)
print('\n Best parameters:')
print(grid.best_params_)
