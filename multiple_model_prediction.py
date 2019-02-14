import missing_value_outlier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

train , test = missing_value_outlier.main()
print(test.isnull().sum())
print(train.isnull().sum())
# train['Loan_Status'].replace('N', 0,inplace=True)
# train['Loan_Status'].replace('Y', 1,inplace=True)
train['Loan_Status']= train['Loan_Status'].map({'N':0,'Y':1}).astype('int')
train['Property_Area']= train['Property_Area'].map({'Semiurban':0,'Urban':1,'Rural':2}).astype('int')
train['Dependents']= train['Dependents'].map({'0':0,'1':1,'2':2,'3+':'3'}).astype('int')

# print(test['LoanAmount'])
# # Feature Scaling

sc = MinMaxScaler()
train[['LoanAmount','ApplicantIncome','CoapplicantIncome']] = sc.fit_transform(train[['LoanAmount','ApplicantIncome','CoapplicantIncome']])
test[['LoanAmount','ApplicantIncome','CoapplicantIncome']] = sc.fit_transform(test[['LoanAmount','ApplicantIncome','CoapplicantIncome']])

# print(train['LoanAmount'])
# print(test)
train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)
X = train.drop('Loan_Status',1)
y = train['Loan_Status']

X=pd.get_dummies(X,prefix_sep='_', drop_first=True)
train=pd.get_dummies(train)
test=pd.get_dummies(test)
validation_size = 0.2
seed = 450



models = []
models.append(('LR', LogisticRegression(random_state=1,solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=1)))
models.append(('NB', GaussianNB()))
models.append(('RFC', RandomForestClassifier(random_state=1, max_depth=10)))
models.append(('SVM', SVC(gamma='auto')))
# print(models)
# evaluate each model in turn
results = []
names = []
for name, model in models:
    # print(name,model)
    for i in range(10,1000,20):
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,
                                                                                        y,
                                                                                        test_size=validation_size,
                                                                                        random_state=i)
        kfold = model_selection.KFold(n_splits=15, random_state=i)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)