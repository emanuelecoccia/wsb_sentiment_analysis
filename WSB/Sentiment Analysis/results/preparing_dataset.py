import pandas as pd
import numpy as np
import yfinance as yf

# the purpose of this script is to prepare our dataset for plotting.
# we want to plot on the same graph the sentiment and the stock price:
# since stocks are not traded on weekends and their sentiment can be null in some days of the week,
# there could be some holes in the datetime (x) variable, therefore
# we will merge those two dataset with an empty dataset of dates
# (so that we won't have missing dates)

df = pd.read_csv('/wsb_dataset_final.csv', sep=';')

# we want to observe the stocks that are more relevant
print(df['stock_symbol'].value_counts()[:30])
stock_list = ['GME', 'AMC', 'CLOV', 'WISH', 'UWMC', 'CLNE', 'NOK', 'SPCE', 'PLTR', 'TSLA', 'WKHS', 'SPY', 'RKT', 'CRSR', 'BB']

df = df[['dt', 'stock_symbol', 'sentiment', 'score', 'upvote_ratio']]
df = df.loc[df['stock_symbol'].isin(stock_list)]

# we have three different time periods: 
# we can create a new label to better segment our data
# I am aware it might seem inefficient, but it is useful to explore the data
def assign_group(row):
  if row['dt'] < '2021-01-01':
    return 3
  if row['dt'] > '2021-01-01' and row['dt'] < '2021-03-01':
    return 1
  if row['dt'] > '2021-03-01':
    return 2

df['group'] = df.apply(assign_group, axis = 1)

# what we are doing here is eliminating the time component and leaving only the dates rounded to the day
df['dt'] = df['dt'].apply(pd.Timestamp).apply(lambda x: x.round(freq = 'D'))

# for clarity purposes
df = df.replace('not bullish', 'not_bullish')

df1 = df.loc[df['group'] == 1].drop(columns = 'group')
df2 = df.loc[df['group'] == 2].drop(columns = 'group')

def prepare_df(df, start_date, end_date, end_date_plus_one):
  
  # we tranform the sentiment variable into a dummy, 
  # after that we multiply it by (num of comments + num of upvotes) 
  # and again by the upvote ratio (ratio of upvotes on total votes)
  # to get a raw 'engagement' metric of the post
  df = pd.get_dummies(df, columns = ['sentiment'])
  df['sentiment_bullish'] = df['sentiment_bullish'] * df['score'] * df['upvote_ratio']
  df['sentiment_not_bullish'] *= df['sentiment_not_bullish'] * df['score'] * df['upvote_ratio']
  df = df.drop(columns = ['score', 'upvote_ratio'])
  
  # we put the data in order, grouping by stock and dates
  df = df.set_index(['stock_symbol', 'dt']).groupby([pd.Grouper(level = 0), pd.Grouper(level = 1)]).sum()
  df = df.reset_index()

  # now we create a dataframe containing only stock symbols and dates, and we merge it with the initial df
  l = []
  for stock in stock_list:
    frame = pd.DataFrame((pd.date_range(start=start_date, end=end_date)), columns = ('dt', ))
    frame['stock_symbol'] = stock
    l.append(frame)
  df_date = pd.concat(l)

  df = pd.merge(df_date, df, how = 'left', on = ['dt', 'stock_symbol']).fillna(0)

  # we download market data and we merge it with the previous df
  hist = yf.download(
    stock_list, start=start_date, end=end_date_plus_one, groupby = 'ticker'
    ).stack().rename_axis(['dt', 'stock_symbol']).reset_index()
  df = pd.merge(df, hist, how = 'left', on = ['dt', 'stock_symbol'])

  # market data also has gaps during the weekends, 
  # so we fill those values with the previous friday's values
  df.fillna(method='ffill', inplace=True)

  return df

df1 = prepare_df(df1, '2021-01-04', '2021-02-03', '2021-02-04')
df2 = prepare_df(df2, '2021-05-11', '2021-06-10', '2021-06-11')

df1.to_csv('/wsb_winter_2021.csv', sep = ';')
df2.to_csv('/wsb_summer_2021.csv', sep = ';')
