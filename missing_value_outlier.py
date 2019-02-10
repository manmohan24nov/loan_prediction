import loan_data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

train = loan_data.main()
print(train.columns)
print(train.isnull().sum())

## take it forword link
##https://trainings.analyticsvidhya.com/courses/course-v1:AnalyticsVidhya+LP101+2018_T1/courseware/1f874ddf371844cab9846b1600be7d60/e683a57fec5447a79ade5ce3a7960056/?child=last
