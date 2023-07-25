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
import time

df = pd.read_csv("data/data_withfeatures.csv")
X = df.drop('Status', axis=1)
y = df['Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

print("xgboost")
start = time.time()
xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.10,score='f1')
xgb.fit(X_train, y_train)
xgb_predict_probs = xgb.predict_proba(X_test)[:, 1]
xgb_fpr, xgb_tpr, xgb_thr = roc_curve(y_test, xgb_predict_probs)
xgb_roc_auc = auc(xgb_fpr, xgb_tpr)
xgb_predictions = xgb.predict(X_test)
print(classification_report(y_test, xgb_predictions))
end = time.time()
print(end - start)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
C = confusion_matrix(y_test, xgb.predict(X_test), labels=[0, 1])
df = pd.DataFrame(C, index=["Repaid", "Late"], columns=["Repaid", "Late"])
df.to_csv("data/confusion_matrix.csv", index=False)
sns.heatmap(df, annot=True, fmt='g', cmap='YlGnBu_r')
plt.show()
