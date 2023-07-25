import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# load the data and check the datatype
loan = pd.read_csv('data/data_cleaned.csv')
print(loan.info())

# UserName startswith BO is the default username, we split the two groups and generate dummies
loan['UserName_default'] = loan['UserName'].apply(lambda x: x.startswith("BO"))
loan.drop('UserName', axis=1, inplace=True)
# print(loan['UserName_default'].value_counts())

# then we deal with the datetime related features, we generate year,month, day of week for each of them
date_features = ["LoanApplicationStartedDate", "LoanDate", "FirstPaymentDate", "MaturityDate_Original"]
loan[date_features] = loan[date_features].apply(pd.to_datetime)
for date_feature in date_features:
    loan[date_feature + '_YEAR'] = loan[date_feature].dt.year
    loan[date_feature + '_MOY'] = loan[date_feature].dt.month
    loan[date_feature + '_DOW'] = loan[date_feature].dt.dayofweek
loan.drop(date_features, axis=1, inplace=True)

# VerificationType,Gender and OccupationArea HomeOwnershipType datatype is float with values like 1.0,2.0
# we convert them to integers
features = ['VerificationType', 'Gender', 'Education', 'MaritalStatus',
            'EmploymentStatus', 'OccupationArea', 'HomeOwnershipType']
for feature in features:
    loan[feature] = loan[feature].astype(int)
    # print(loan[feature].value_counts())

# we change the Status in to 0 and 1, 0 for Repaid and 1 for Late
loan['Status'] = loan['Status'].apply(lambda x: 1 if x == 'Late' else 0)

# We generate dummies for all the categorical variables
categorical_features = ['NewCreditCustomer', 'VerificationType', 'ApplicationSignedHour', 'ApplicationSignedWeekday',
                        'LanguageCode', 'Gender', 'Country', 'UseOfLoan', 'Education', 'MaritalStatus',
                        'EmploymentStatus', 'EmploymentDurationCurrentEmployer', 'OccupationArea', 'HomeOwnershipType',
                        'UserName_default', 'LoanApplicationStartedDate_YEAR',
                        'LoanApplicationStartedDate_MOY', 'LoanApplicationStartedDate_DOW', 'LoanDate_YEAR',
                        'LoanDate_MOY', 'LoanDate_DOW', 'FirstPaymentDate_YEAR', 'FirstPaymentDate_MOY',
                        'FirstPaymentDate_DOW', 'MaturityDate_Original_YEAR', 'MaturityDate_Original_MOY',
                        'MaturityDate_Original_DOW']
loan = pd.get_dummies(data=loan, columns=categorical_features, drop_first=True)
print(loan.info())

# save the dataset after feature engineering to csv file
# and we will train and test the data later
loan.to_csv("data/data_withfeatures.csv", index=False)
