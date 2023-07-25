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
loan = loan[loan["Status"] != "Current"]
lists = ["UserName",
         "NewCreditCustomer",
         "LoanApplicationStartedDate",
         "LoanDate",
         "FirstPaymentDate",
         "MaturityDate_Original",
         "VerificationType",
         "ApplicationSignedHour",
         "ApplicationSignedWeekday",
         "LanguageCode",
         "Age",
         "Gender",
         "Country",
         "AppliedAmount",
         "Amount",
         "Interest",
         "LoanDuration",
         "MonthlyPayment",
         "UseOfLoan",
         "Education",
         "MaritalStatus",
         "EmploymentStatus",
         "EmploymentDurationCurrentEmployer",
         "OccupationArea",
         "HomeOwnershipType",
         "IncomeFromPrincipalEmployer",
         "IncomeFromPension",
         "IncomeFromFamilyAllowance",
         "IncomeFromSocialWelfare",
         "IncomeFromLeavePay",
         "IncomeFromChildSupport",
         "IncomeOther",
         "IncomeTotal",
         "ExistingLiabilities",
         "LiabilitiesTotal",
         "RefinanceLiabilities",
         "DebtToIncome",
         "FreeCash",
         "MonthlyPaymentDay",
         "Status",
         "NoOfPreviousLoansBeforeLoan",
         "AmountOfPreviousLoansBeforeLoan"
         ]
loan = loan[lists]
loan.dropna(inplace=True)
loan.to_csv("data/data_cleaned.csv", index=False)
