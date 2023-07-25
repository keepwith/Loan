import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import time

df = pd.read_csv("data/data_withfeatures.csv")
X = df.drop('Status', axis=1)
y = df['Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# def XGB_model(X_train, y_train, X_test, y_test):
params = {
    'n_estimators': [300, 350, 400, 450, 500, 550, 600, 650, 700],
    'learning_rate': [0.01, 0.05, 0.10, 0.15, 0.20],
    'max_depth': [5, 6, 7, 8]
}

# Logistic Regression
print("Logistic Regression")
start = time.time()
lr = LogisticRegression(solver='liblinear', max_iter=5000)
lr.fit(X_train, y_train)
lr_predict_probs = lr.predict_proba(X_test)[:, 1]
lr_fpr, lr_tpr, lr_thr = roc_curve(y_test, lr_predict_probs)
lr_roc_auc = auc(lr_fpr, lr_tpr)
lr_predictions = lr.predict(X_test)
print(classification_report(y_test, lr_predictions))
end = time.time()
print(end - start)

# Random Forest
print("Random Forest")
start = time.time()
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_predict_probs = rfc.predict_proba(X_test)[:, 1]
rfc_fpr, rfc_tpr, rfc_thr = roc_curve(y_test, rfc_predict_probs)
rfc_roc_auc = auc(rfc_fpr, rfc_tpr)
rfc_predictions = rfc.predict(X_test)
print(classification_report(y_test, rfc_predictions))
end = time.time()
print(end - start)

# K nearest neighbors
print("K Nearest Neighbors")
start = time.time()
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_predict_probs = knn.predict_proba(X_test)[:, 1]
knn_fpr, knn_tpr, knn_thr = roc_curve(y_test, knn_predict_probs)
knn_roc_auc = auc(knn_fpr, knn_tpr)
knn_predictions = knn.predict(X_test)
print(classification_report(y_test, knn_predictions))
end = time.time()
print(end - start)

# Naive Bayes
print("Naive Bayes")
start = time.time()
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_predict_probs = nb.predict_proba(X_test)[:, 1]
nb_fpr, nb_tpr, nb_thr = roc_curve(y_test, nb_predict_probs)
nb_roc_auc = auc(nb_fpr, nb_tpr)
nb_predictions = nb.predict(X_test)
print(classification_report(y_test, nb_predictions))
end = time.time()
print(end - start)

# xgboost
print("XGBoost")
start = time.time()
xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.10)
xgb.fit(X_train, y_train)
xgb_predict_probs = xgb.predict_proba(X_test)[:, 1]
xgb_fpr, xgb_tpr, xgb_thr = roc_curve(y_test, xgb_predict_probs)
xgb_roc_auc = auc(xgb_fpr, xgb_tpr)
xgb_predictions = xgb.predict(X_test)
print(classification_report(y_test, xgb_predictions))
end = time.time()
print(end - start)

# Gradient Boosting Tree
print("Gradient Boosting Tree")
start = time.time()
gdbt = GradientBoostingClassifier()
gdbt.fit(X_train, y_train)
gdbt_predict_probs = gdbt.predict_proba(X_test)[:, 1]
gdbt_fpr, gdbt_tpr, gdbt_thr = roc_curve(y_test, gdbt_predict_probs)
gdbt_roc_auc = auc(gdbt_fpr, gdbt_tpr)
gdbt_predictions = gdbt.predict(X_test)
print(classification_report(y_test, gdbt_predictions))
end = time.time()
print(end - start)

# SVM
print("Support Vector Machine")
start = time.time()
svm = SVC(probability=True)
svm.fit(X_train, y_train)
svm_predict_probs = svm.predict_proba(X_test)[:, 1]
svm_fpr, svm_tpr, svm_thr = roc_curve(y_test, svm_predict_probs)
svm_roc_auc = auc(svm_fpr, svm_tpr)
svm_predictions = svm.predict(X_test)
print(classification_report(y_test, svm_predictions))
end = time.time()
print(end - start)

plt.figure()
plt.plot(lr_fpr, lr_tpr, color='darkorange', lw=2, label='Logistic Regression (area = %0.2f)' % lr_roc_auc)
plt.plot(rfc_fpr, rfc_tpr, color='darkgreen', lw=2, label='Random Forest (area = %0.2f)' % rfc_roc_auc)
plt.plot(knn_fpr, knn_tpr, color='darkred', lw=2, label='K-Nearest Neighbors (area = %0.2f)' % knn_roc_auc)
plt.plot(nb_fpr, nb_tpr, color='blue', lw=2, label='Naive Bayes (area = %0.2f)' % nb_roc_auc)
plt.plot(xgb_fpr, xgb_tpr, color='red', lw=2, label='Xgboost (area = %0.2f)' % xgb_roc_auc)
plt.plot(gdbt_fpr, gdbt_tpr, color='purple', lw=2, label='Gradient Boosting Tree (area = %0.2f)' % gdbt_roc_auc)
plt.plot(svm_fpr, svm_tpr, color='yellow', lw=2, label='Support Vector Machine (area = %0.2f)' % svm_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
