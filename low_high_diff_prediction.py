from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error
from sklearn.svm import SVR
from sklearn import model_selection

import test_concept

data = test_concept.main()
data = data.dropna()
sc = MinMaxScaler()

X = data.drop(['Date','WAP','No.of Shares','No. of Trades','Total Turnover (Rs.)',
               'Deliverable Quantity','% Deli. Qty to Traded Qty','new_date','close_high_diff'],axis=1)

Y = data['close_high_diff']
for i in ['month','week_of_year','day','year','month_and_day','month_and_week_of_year','week_of_year_and_day']:
    X[i] = X[i].astype('int')


X['day_of_week'] = X['day_of_week'].map({'Tuesday':'0.2', 'Wednesday':'0.3',
                                         'Thursday':'0.4', 'Friday':'0.5', 'Monday':'0.1'}).astype('float')

# X = sc.fit_transform(X)
# Y = sc.fit_transform(Y)
model = SVR(C=10,gamma=0.1)
for i in range(1,1000,50):
    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(X,Y,test_size=0.2,random_state=i)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_validation)
    print(list(y_pred))
    print(list(y_validation))
    print('random state', i)
    accuracy = mean_absolute_error(y_validation, y_pred)
    print(accuracy * 100)




# Cs = [0.001,0.01,0.1,1,10,100]
# gammas = [0.001,0.01,0.1,1,10]
# param_grid = {'C':Cs,'gamma':gammas}
#
# grid_selection = model_selection.GridSearchCV(model, param_grid=param_grid,scoring = 'neg_mean_squared_error', cv=150)
# grid_selection.fit(x_train,y_train)
# best_param = grid_selection.best_params_
# print(best_param)
# best_result = grid_selection.best_score_
# print(best_result)


