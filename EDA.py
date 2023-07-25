import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

types = {'NrOfDependants': object, 'WorkExperience': object, 'Rating_V0': object,
         'Rating_V1': object,
         'Rating_V2': object, 'CreditScoreEsEquifaxRisk': object, 'CreditScoreFiAsiakasTietoRiskGrade': object}
loan = pd.read_csv('data/LoanData_Bondora.csv', dtype=types)
loan.rename(columns={"PreviousEarlyRepaymentsBefoleLoan": "PreviousEarlyRepaymentsBeforeLoan"})
print(loan.iloc[:, 95:111].info())
# print(loan.head())
# print(loan["PreviousRepaymentsBeforeLoan"].isna())
print(loan[loan["PreviousRepaymentsBeforeLoan"].isna()]["AmountOfPreviousLoansBeforeLoan"].value_counts())

# drop_list = ["ReportAsOfEOD", "LoanId", "LoanNumber"]
# loan.drop(drop_list, axis=1, inplace=True)
# loan["ContractEndDate"] = loan["ContractEndDate"].apply(pd.to_datetime)
# loan["ContractEndDate_year"] = loan["ContractEndDate"].dt.year
# sns.countplot(data=loan[loan["Status"] == "Late"], x="ContractEndDate_year")
# sns.displot(data=loan, x="PreviousEarlyRepaymentsBeforeLoan")
# plt.show()
