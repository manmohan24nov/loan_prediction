import missing_value_outlier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, log_loss
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

train , test = missing_value_outlier.main()
train['Loan_Status']= train['Loan_Status'].map({'N':0,'Y':1}).astype('int')
train['Property_Area']= train['Property_Area'].map({'Semiurban':0,'Urban':0.1,'Rural':0.2}).astype('float')
train['Dependents']= train['Dependents'].map({'0':0,'1':0.1,'2':0.2,'3+':0.3}).astype('float')

train['total_income']=train['ApplicantIncome']+train['CoapplicantIncome']
train = train.drop(['CoapplicantIncome','ApplicantIncome'],axis=1)
test['total_income']=test['ApplicantIncome']+test['CoapplicantIncome']
test = test.drop(['CoapplicantIncome','ApplicantIncome'],axis=1)


sc = MinMaxScaler()
train[['LoanAmount','total_income']] = sc.fit_transform(train[['LoanAmount','total_income']])
test[['LoanAmount','total_income']] = sc.fit_transform(test[['LoanAmount','total_income']])


# print(train['LoanAmount'])
# print(test)
train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)


X = train.drop('Loan_Status',1)
y = train['Loan_Status']

X=pd.get_dummies(X,prefix_sep='_', drop_first=True)

train=pd.get_dummies(train,prefix_sep='_', drop_first=True)
test=pd.get_dummies(test)

new_data_x = X.values
Y = y.values
print(new_data_x,Y)

X = new_data_x[:,:].astype(float)

# Label encode Class (Species)
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# One Hot Encode
y_dummy = np_utils.to_categorical(encoded_Y)

validation_size = 0.2
seed = 1710


def deepml_model(optimizer):
    # Model Creation
    deepml = Sequential()
    deepml.add(Dense(8, input_dim=10, activation='relu'))
    deepml.add(Dense(2, activation='softmax'))
    # Model Compilation
    deepml.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    return deepml

estimate = KerasClassifier(build_fn=deepml_model)
# k_fold = KFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimate, X, Y, cv=k_fold)
# print("Model: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

parameters = {'batch_size': [20,25],
              'epochs': [ 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = estimate,
                           param_grid = parameters,
                           scoring = make_scorer(log_loss, needs_proba=True, labels=y_dummy),
                           cv = 3)
grid_search = grid_search.fit(X,y_dummy)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_accuracy,best_parameters)
