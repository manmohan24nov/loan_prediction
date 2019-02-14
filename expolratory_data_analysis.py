import loan_data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data_set = loan_data.main()
print(data_set.columns)
# print(data_set)
##############Target Variable
####We will first look at the target variable, i.e., Loan_Status. As it is a categorical variable, let us look at its frequency table,
##  percentage distribution and bar plot.
##### Frequency table of a variable will give us the count of each category in that variable.
# Normalize can be set to True to print proportions instead of number
# print(data_set['Loan_Status'].value_counts(normalize=True))
# plt.figure(1)

##Now lets visualize each variable separately. Different types of variables are Categorical, ordinal and numerical.

##Categorical features: These features have categories (Gender, Married, Self_Employed, Credit_History, Loan_Status)
##Ordinal features: Variables in categorical features having some order involved (Dependents, Education, Property_Area)
##Numerical features: These features have numerical values (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term)

##Independent Variable (Categorical)
plt.subplot(221)
# data_set['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender')
# plt.subplot(222)
# data_set['Married'].value_counts(normalize=True).plot.bar(title= 'Married')
# plt.subplot(223)
# data_set['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')
# plt.subplot(224)
# data_set['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')


#####Independent Variable (Ordinal)
# plt.figure(1)
# plt.subplot(131)
# data_set['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents')
#
# plt.subplot(132)
# data_set['Education'].value_counts(normalize=True).plot.bar(title= 'Education')


# plt.subplot(133)
# data_set['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')

####Independent Variable (Numerical)

# plt.figure(1)
# plt.subplot(121)
# sns.distplot(data_set['ApplicantIncome']);
#
# plt.subplot(122)
# data_set['ApplicantIncome'].plot.box(figsize=(16,5))
#
# data_set.boxplot(column='ApplicantIncome', by = 'Education')
# plt.suptitle("")
#
# plt.figure(3)
# plt.subplot(121)
# sns.distplot(data_set['CoapplicantIncome']);
#
# plt.subplot(122)
# data_set['CoapplicantIncome'].plot.box(figsize=(16,5))
#
# plt.figure(5)
# plt.subplot(121)
# df=data_set.dropna()
# sns.distplot(df['LoanAmount']);
#
# plt.subplot(122)
# data_set['LoanAmount'].plot.box(figsize=(16,5))
print(data_set[data_set['Loan_Status']].corr())
plt.show()
