import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from xgboost import plot_importance
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/data_withfeatures.csv")
X = df.drop('Status', axis=1)
y = df['Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.10)
xgb.fit(X_train, y_train)
xgb_predict_probs = xgb.predict_proba(X_test)[:, 1]
xgb_fpr, xgb_tpr, xgb_thr = roc_curve(y_test, xgb_predict_probs)
xgb_roc_auc = auc(xgb_fpr, xgb_tpr)
fig = plt.figure(figsize=(20, 4))
results = pd.DataFrame()
results['columns'] = X.columns
results['importances'] = xgb.feature_importances_
results.sort_values(by='importances', ascending=False, inplace=True)
results[:20]
sns.barplot(x=results[:20]['importances'], y=results[:20]['columns'])
plt.xlabel("importance")
plt.ylabel("features")
plt.show()
