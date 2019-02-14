import loan_data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

train = loan_data.main()
# print(train.columns)
print(train.isnull().sum())


test = pd.read_csv('test_Y3wMUE5_7gLdaTN.csv',header=0,sep=',')

## take it forword link
##https://trainings.analyticsvidhya.com/courses/course-v1:AnalyticsVidhya+LP101+2018_T1/courseware/1f874ddf371844cab9846b1600be7d60/e683a57fec5447a79ade5ce3a7960056/?child=last

def main():
    # changes in train data
    train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
    train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
    train['Married'].fillna(train['Married'].mode()[0], inplace=True)
    train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
    train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
    train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
    train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

    # changes in test data
    test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
    test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
    test['Married'].fillna(train['Married'].mode()[0], inplace=True)
    test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
    test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
    test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
    test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
    return train,test

if __name__ == '__main__':
    main()

