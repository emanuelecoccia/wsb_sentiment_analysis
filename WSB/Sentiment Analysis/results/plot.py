import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML

# plot settings
display(HTML("<style>div.output_scroll { height: 31em; }</style>"))
plt.rcParams['figure.figsize'] = 18, 8
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 2

df1 = pd.read_csv('/content/wsb_winter_2021.csv', sep=';')
df2 = pd.read_csv('/content/wsb_summer_2021.csv', sep=';')

# we want to plot the stock sentiment and the stock price at the same time, 
# so we use 2 subplots
def create_plot(h, stock, date):
  fig, ax1 = plt.subplots()

  ax1.set_xlabel('days', fontsize = 16)
  ax1.set_ylabel('interactions', fontsize = 16)
  ax1.stackplot(h.dt, [h.sentiment_not_bullish, h.sentiment_bullish], labels = ['not bullish', 'bullish'], colors = ['#565656', '#fcbf01'])
  ax1.set_title(f'{stock}, {date}', fontsize = 14)
  ax1.set_facecolor('#1a192e')
  ax1.figure.patch.set_facecolor('#e8e8e8')
  ax1.figure.autofmt_xdate()

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  ax2.set_ylabel('price', fontsize = 16)
  ax2.plot(h.dt, h.Close, label = 'price', color='#d9d9d9')
  ax2.margins(y=0.05)

  # we need this snippet of code to plot a common legend, 
  # since we have two different subplots
  lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
  lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
  ax1.legend(lines, labels, loc = 'upper left', fontsize=18)

  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  plt.show()
  
stock_list = ['GME', 'AMC', 'CLOV', 'WISH', 'UWMC', 'CLNE', 'NOK', 'SPCE', 'PLTR', 'TSLA', 'WKHS', 'SPY', 'RKT', 'CRSR', 'BB']

# finally we plot our results
for stock in stock_list:
  h = df1.loc[df1['stock_symbol'] == stock]
  create_plot(h, stock, 'January 2021')

for stock in stock_list:
  h = df2.loc[df2['stock_symbol'] == stock]
  create_plot(h, stock, 'June 2021')
