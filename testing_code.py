import loan_data
import matplotlib.pyplot as plt
import pandas as pd



train = loan_data.main()
# Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
# # Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
# # plt.show()
#
# print(Dependents)
# print(Dependents.div(Dependents.sum(1).astype(float), axis=0))
#
#
# fig, axes = plt.subplots(nrows=2, ncols=2)
#
# # fig,ax = plt.subplot(2,1)
# Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,ax=axes[0,0])
# # ax.bar(Dependents.div(Dependents.sum(1).astype(float), axis=0), stacked=True)
# # plt.show()
# # ax2 = plt.subplot(1, 2, 2)
# Dependents.plot(kind="bar", stacked=True,ax=axes[0,1])
#
# plt.show()

print(train[(train['ApplicantIncome'] != None) & (train['CoapplicantIncome']==None)])
print(train['CoapplicantIncome'].isnull())