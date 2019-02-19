import missing_value_outlier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

train , test = missing_value_outlier.main()
# print(test.isnull().sum())
# print(train.isnull().sum())
train['Loan_Status']= train['Loan_Status'].map({'N':0,'Y':1}).astype('int')
train['Property_Area']= train['Property_Area'].map({'Semiurban':0,'Urban':0.1,'Rural':0.2}).astype('float')
train['Dependents']= train['Dependents'].map({'0':0,'1':0.1,'2':0.2,'3+':0.3}).astype('float')

train['total_income']=train['ApplicantIncome']+train['CoapplicantIncome']
# train = train.drop(['CoapplicantIncome','ApplicantIncome','Property_Area','Self_Employed','Gender','Dependents','Education','Loan_Amount_Term','LoanAmount','Married','total_income'],axis=1)
test['total_income']=test['ApplicantIncome']+test['CoapplicantIncome']
# test = test.drop(['CoapplicantIncome','ApplicantIncome','Property_Area','Self_Employed','Gender','Dependents','Education','Loan_Amount_Term','LoanAmount','Married','total_income'],axis=1)

# train = train.drop('Self_Employed', axis=1)
# test = test.drop('Self_Employed', axis=1)


# print(train[['Property_Area','Dependents']].describe())
# print(train.Dependents.value_counts())
# train['Loan_Status'].replace('Y', 1,inplace=True)
# raw_data['class'] = raw_data['class'].map({'Iris-setosa': 1, 'Iris-versicolor' : 2, 'Iris-virginica' : 3})

# print(test['LoanAmount'])
# # Feature Scaling
#
# sc = MinMaxScaler()
# train[['total_income']] = sc.fit_transform(train[['total_income']])
# test[['total_income']] = sc.fit_transform(test[['total_income']])


# print(train['LoanAmount'])
# print(test)

train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)
X = train.drop('Loan_Status',1)
y = train['Loan_Status']

X=pd.get_dummies(X,prefix_sep='_')

train=pd.get_dummies(train,prefix_sep='_')
test=pd.get_dummies(test)

print(train)
validation_size = 0.2
seed = 1750
pd.set_option('display.max_columns',12)
# print(X.dtypes)
#
print(train[train['Loan_Status']==1].corr())


model = RandomForestClassifier(random_state=1, max_depth=10)
for i in range(10,2000,10):
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,
                                                                    y,
                                                                    test_size=validation_size,
                                                                    random_state=i)

    model.fit(X_train,Y_train)
    y_pred = model.predict(X_validation)
    accuracy = accuracy_score(Y_validation,y_pred)
    print(i)
    print(accuracy*100)


importances=pd.Series(model.feature_importances_, index=X.columns)
importances.plot(kind='barh', figsize=(12,8))
plt.show()
#
# from sklearn.metrics import confusion_matrix
#
# cm=confusion_matrix(Y_validation ,y_pred)
#
# print(cm)
#
# cm = pd.crosstab(Y_validation ,y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
# print(cm)
# pred_test = model.predict(test)
# submission=pd.read_csv("Sample_Submission_ZAuTl8O_FK3zQHh.csv")
# submission['Loan_Status']=pred_test
# submission['Loan_ID']=test_original['Loan_ID']
# submission['Loan_Status'].replace(0, 'N',inplace=True)
# submission['Loan_Status'].replace(1, 'Y',inplace=True)
# pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')

