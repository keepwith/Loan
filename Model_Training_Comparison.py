import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import time
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

df = pd.read_csv("data/data_withfeatures.csv")
X = df.drop('Status', axis=1)
y = df['Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

results = []
names = []

models = []
models.append(('LR', LogisticRegression(solver='liblinear', max_iter=5000, random_state=101)))
models.append(('RFC', RandomForestClassifier(n_estimators=100, random_state=101)))
models.append(('KNC', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('XGBoost', XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.10)))
models.append(('GDBT', GradientBoostingClassifier()))
models.append(('SVM', SVC()))
for name, model in models:
    start = time.time()
    cv_results = cross_val_score(model, X_train, y_train, cv=10, scoring='f1')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    end = time.time()
    print(end - start)
fig = plt.figure()
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
