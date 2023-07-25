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

print("XGBoost")
start = time.time()
xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.10)
xgb.fit(X_train, y_train)
xgb_predictions = xgb.predict(X_test)
print(type(xgb_predictions))
predict_data = pd.concat([X_test, y_test, pd.Series(xgb_predictions).rename('y_predict')],
                         ignore_index=True, axis=1)
print(predict_data)
predict_data.to_csv("data/predict_data.csv", index=False)
end = time.time()
print(end - start)

types = {'NrOfDependants': object, 'WorkExperience': object, 'Rating_V0': object,
         'Rating_V1': object,
         'Rating_V2': object, 'CreditScoreEsEquifaxRisk': object, 'CreditScoreFiAsiakasTietoRiskGrade': object}
loan = pd.read_csv('data/LoanData_Bondora.csv', dtype=types)
loan.rename(columns={"PreviousEarlyRepaymentsBefoleLoan": "PreviousEarlyRepaymentsBeforeLoan"})
loan = loan[loan["Status"] != "Current"]
loan.rename({'Status': 'raw_Status'}, axis=1, inplace=True)
full_data = pd.concat([loan, predict_data], sort=False)
# print(loan[["Age", "AppliedAmount", "Amount", "Interest", "LoanDuration", "MonthlyPayment",
#             "IncomeFromPrincipalEmployer", "LiabilitiesTotal", "IncomeTotal", "ExistingLiabilities",
#             "DebtToIncome", "FreeCash", "MonthlyPaymentDay", "RefinanceLiabilities",
#             "IncomeFromPension", "AmountOfPreviousLoansBeforeLoan", "NoOfPreviousLoansBeforeLoan",
#             "IncomeOther", "IncomeFromChildSupport", "IncomeFromLeavePay", "IncomeFromSocialWelfare",
#             "IncomeFromFamilyAllowance"]].value_counts())
print(full_data)
print(len(full_data))
