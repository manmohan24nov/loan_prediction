import stock_raw_data
import matplotlib.pyplot as plt

stock_name_local = 'reliance_comm'
stock_data = stock_raw_data.stock_data(stock_name_local).stock_data_func()
stock_data = stock_data[(stock_data['new_date'] > '2018-01-01') & (stock_data['new_date'] <= '2018-12-31')]
# print(stock_data.columns)

# ---------------------------------check for high and low frequency prediction-------------------------------------
def main():

    stock_data_new = stock_data.copy()
    stock_data_new['day_of_week'] = stock_data_new['new_date'].dt.day_name()
    stock_data_new['month'] = stock_data_new['new_date'].dt.month.astype(str).str.zfill(2)
    stock_data_new['week_of_year'] = stock_data_new['new_date'].dt.week.astype(str).str.zfill(2)
    stock_data_new['day'] = stock_data_new['new_date'].dt.day.astype(str).str.zfill(2)
    stock_data_new['year'] = stock_data_new['new_date'].dt.year.astype(str)
    stock_data_new['month_and_day'] = stock_data_new[['month','day']].apply(lambda x: ''.join(x), axis=1)
    stock_data_new['month_and_week_of_year'] = stock_data_new[['month','week_of_year']].apply(lambda x: ''.join(x), axis=1)
    stock_data_new['week_of_year_and_day'] = stock_data_new[['week_of_year','day']].apply(lambda x: ''.join(x), axis=1)

    # stock_data_new = stock_data_new[stock_data_new['Spread High-Low'] > 2]
    stock_data_new['Spread High-Low'] = stock_data_new['Spread High-Low'].apply(lambda x: round(x))
    # print(stock_data_new['Spread High-Low'])
    stock_data_new['close_low_difference'] = stock_data_new['Close Price'].shift(1) - stock_data_new['Low Price']
    stock_data_new['close_high_diff'] = stock_data_new['Close Price'].shift(1) - stock_data_new['High Price']
    # print(stock_data_new[['close_low_difference','close_high_diff','Close Price']].describe())
    # stock_data_new.groupby(['month_and_week_of_year']).agg({'close_low_difference':'median','close_high_diff':'median','Spread High-Low':'median'}).to_csv('monthly2019.csv',header=True)
    # abc = stock_data_new.sort_values('year')
    # plt.scatter(abc['year'],abc['Spread High-Low'])
    # stock_data_new['close_low_difference'].plot(grid=True)
    # plt.scatter(stock_data_new['year'],stock_data_new['close_high_diff'])
    # print(stock_data_new.head(10))
    return stock_data_new

if __name__ == '__main__':
    main()
# ----------------------------------------end of this research-----------------------------------------------


# stock_data['difference'] = (stock_data['No. of Trades'].shift(-1)-stock_data['No. of Trades'])/stock_data['No. of Trades'] * 100
# print(stock_data.loc[stock_data['difference'].idxmin()])
# print(stock_data.loc[stock_data['difference']> 10])
# plt.scatter(stock_data.index,stock_data['difference'])
plt.show()

