
# coding: utf-8

# In[56]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore') # Warnings were getting annoying.
from bokeh.events import ButtonClick
from bokeh.layouts import column, row, widgetbox, Spacer
from bokeh.models.widgets import Button, Select, CheckboxButtonGroup, DatePicker
from bokeh.plotting import figure, output_file, show, ColumnDataSource, curdoc
from bokeh.models import HoverTool, CustomJS, Legend
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.io import output_file, show
from math import pi
from bokeh.embed import components


# In[57]:

def ichimoku_plot(df):
    '''Computes the Ichimok trend identification system.'''
    # This plot has 5 components to it.
    high_prices = df['High']
    close_prices = df['Close']
    low_prices = df['Low']
    dates = df.index

    # Ichimoku (Conversion Line): (9-period high + 9-period low)/2
    nine_period_high = df['High'].rolling(window=9, center=False).max() # Usually window is 9 days.
    nine_period_low = df['Low'].rolling(window=9, center=False).max()
    ichimoku = (nine_period_high + nine_period_low) /2
    df['tenkan_sen'] = ichimoku

    # Kijun-Sen (Base line): (26-period high + 26-period low)/2)
    period26_high = high_prices.rolling(window=26, center=False).max() # Window normally 26 days.
    period26_low = low_prices.rolling(window=26, center=False).min()
    df['kijun_sen'] = (period26_high + period26_low) / 2

    # Senkou Span A (Leading span A): (Base line + Conversion line) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    # Senkou Span B (Leading span B): (52-period high + 52-period low)/2
    period52_high = high_prices.rolling(window=52, center=False).max()
    period52_low = low_prices.rolling(window=52, center=False).min()
    df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)

    # Chikou Span (Lagging span): Closing price of last 22 periods.
    df['chikou_span'] = close_prices.shift(-22)

    return df[df.columns[6:]]


# In[58]:

def reverse_date(df, remove_date="Yes", ich_plot="Yes"):
    '''Reverses the dataset so it is in chronological order. Optional to remove date column and set as index.'''
    final_data = df.reindex(index=df.index[::-1])

    # Fixing incorrect dates.
    for i in range(0, len(df)):
        if "v" in df.ix[i,'Date']:
            new_date = df.ix[i,'Date'].replace('v', '/')
            final_data.ix[i,'Date'] = new_date

    if remove_date == "Yes":
        # Need to reverse order of dataframe.
        final_data = final_data.set_index(final_data['Date'])
        del final_data['Date']

    # Convert dates to datetime format.
    final_data['Date'] = pd.to_datetime(final_data['Date'], dayfirst = True)

    if ich_plot == "Yes":
        ichimoku_plot(final_data) # Append ichimoku columns.

    return final_data


# In[59]:

def compute_metrics(name, data, rf, mar, market):
    '''Computes 4 metrics of the Cryptocurrency and returns Pandas dataframe.'''
    metrics = []

    metrics.append(calc_sharpe(data, rf))
    metrics.append(calc_sortino(data, rf, mar))
    metrics.append(calc_treynor(data, market))
    metrics.append(calc_info(data, market))

    # Formats everything into a Pandas Dataframe to return.
    metric_names = ["Sharpe Ratio", "Sortino", "Treynor", "Info Ratio"]
    results = pd.DataFrame(data = metrics)
    results.columns = metric_names
    results.set_index(name)

    return results


# In[60]:

def scale_volume(dataframe, scale_factor):
    '''Append a column of volumes scaled down by specified factor'''
    dataframe['Scaled Volume'] = dataframe['Volume']/scale_factor
    return dataframe


# In[61]:

def sma_plot(df, window):
    '''Computes simple moving average.'''
    rolling = df['Close'].rolling(window=window) # Window tells us how many days average to take.
    return rolling.mean()


# In[62]:

def bollinger_plot(df, window, num_sd):
    '''Computes Bollinger bands depending on number of standard deviation and window.'''
    rolling_mean = df['Close'].rolling(window).mean() # Window should be same as SMA.
    rolling_std = df['Close'].rolling(window).std()

    bollinger = pd.DataFrame(data=None)
    bollinger['Rolling Mean'] = rolling_mean
    bollinger['Bollinger High'] = rolling_mean + (rolling_std * num_sd)
    bollinger['Bollinger Low'] = rolling_mean - (rolling_std * num_sd)

    return bollinger


# In[63]:


# Reading data
btc = pd.read_csv('https://raw.githubusercontent.com/chrishyland/Crypto-position/master/bitcoin_final.csv')
df = reverse_date(btc, "No")
df = scale_volume(df, 3000000)

ripple = pd.read_csv('https://raw.githubusercontent.com/chrishyland/Crypto-position/master/ripple_final.csv')
df_rip = reverse_date(ripple, "No")
df_rip = scale_volume(df_rip, 10000000000)

ethereum = pd.read_csv('https://raw.githubusercontent.com/chrishyland/Crypto-position/master/ethereum_final.csv')
df_eth = reverse_date(ethereum, "No")
df_eth = scale_volume(df_eth, 30000000)

# For market index.
df_market = pd.read_csv('https://raw.githubusercontent.com/chrishyland/Crypto-position/master/crix.csv')
df_market.columns = ['Date', 'Close'] # Rename column.
df_market.reindex(index=df_market.index[::-1])
df_market['Date'] = pd.to_datetime(df_market['Date'], dayfirst = True)


# In[65]:

'''Constructing top candlestick chart with ichimoku plot.'''
inc = df.Close > df.Open
dec = df.Open > df.Close
sourceInc_top = ColumnDataSource(ColumnDataSource.from_df(df.loc[inc]))
sourceDec_top = ColumnDataSource(ColumnDataSource.from_df(df.loc[dec]))
source_top = ColumnDataSource(data = df)

inc1 = df_rip.Close > df_rip.Open
dec1 = df_rip.Open > df_rip.Close
sourceInc_bottom = ColumnDataSource(ColumnDataSource.from_df(df_rip.loc[inc1]))
sourceDec_bottom = ColumnDataSource(ColumnDataSource.from_df(df_rip.loc[dec1]))
source_bottom = ColumnDataSource(data = df_rip)


# In[66]:


def make_plot(sourceInc, sourceDec, source, df):
    '''Construct top and bottom candle + ichimoku plot'''
    w = 12*60*60*1000 # Half day in ms.

    # Display last 6 months by default.
    df_6m = df.iloc[-180:,]['Date']

    TOOLS = "pan, wheel_zoom, xbox_select, reset, save"
    p = figure(x_axis_type = "datetime", tools = TOOLS,
               plot_width = 1200, plot_height = 450, x_range = (df_6m.min(), df_6m.max()), active_drag = "xbox_select")
    p.xaxis.major_label_orientation = pi/4
    p.grid.grid_line_alpha = 0.30

    # Construct increasing and decreasing lines.
    seg1 = p.segment('Date', 'High', 'Date', 'Low', color='#17BECF', source = sourceInc)
    seg2 = p.segment('Date', 'High', 'Date', 'Low', color='#FF7777', source = sourceDec)

    # Construct increasing and decreasing bars.
    bar1 = p.vbar('Date', w, 'Open', 'Close', fill_color='#17BECF', line_color='#17BECF', source = sourceInc)
    bar2 = p.vbar('Date', w, 'Open', 'Close', fill_color='#FF7777', line_color='#FF7777', source = sourceDec)

    #Adding hover tool feature.
    hover = HoverTool(
        renderers = [seg1, seg2, bar1, bar2],
        tooltips = [
           ('Date', '@Date{%F}'),
           ('Open', '$@Open{%0.2f}'),
           ('High', '$@High{%0.2f}'),
           ('Low', '$@Low{%0.2f}'),
           ('Close', '$@Close{%0.2f}'),
           ('Volume', '@Volume{0.00 a}'),
        ],
        formatters = {
            'Date' : 'datetime',
            'Open' : 'printf',
            'High' : 'printf',
            'Low' : 'printf',
            'Close': 'printf',
        },
        mode = 'vline'
    )
    p.add_tools(hover)

    # Add line render to display ichimoku plot.
    r1 = p.line('Date', 'tenkan_sen', line_width = 1, color = '#92FFB4', legend = "Tenken-sen",
                muted_alpha = 0.20,  source = source)
    r2 = p.line('Date', 'kijun_sen', line_width = 1, color = '#92B2FF', legend = "Kijun-sen",
                muted_alpha = 0.20, source = source)
    r3 = p.line('Date', 'senkou_span_a', line_width = 1, color = '#98D4FD', legend = "Senkou Span A",
                muted_alpha = 0.20, source = source)
    r4 = p.line('Date', 'senkou_span_b', line_width = 1, color = '#F7B0B6', legend = "Senkou Span B",
                muted_alpha = 0.20, source = source)
    r5 = p.line('Date', 'chikou_span', line_width = 1, color = '#6878FB', legend = "Chikou Span",
                muted_alpha = 0.20, source = source)

    # Add bar render to display scaled down volume. TODO: make generalised scale for all currencies.
    r6 = p.vbar('Date', w, 'Scaled Volume', 0, color='#5DE0F6', source = source)

    p.legend.location = "top_left"
    p.legend.click_policy="mute"
    p.legend.orientation = "horizontal"

    renders = [r1, r2, r3, r4, r5, r6]
    return p, renders


# In[67]:

def fill_area(p, df):
    '''Fill area between senkou span A and B.'''
    index = 0
    index_a = np.argwhere(np.isnan(df['senkou_span_a'].values)).max()
    index_b = np.argwhere(np.isnan(df['senkou_span_b'].values)).max()
    if index_b > index_a:
        index = index_b + 1
    else:
        index = index_a + 1
    dates = df['Date'].values[index:]
    senkou_span_a = df['senkou_span_a'].values[index:]
    senkou_span_b = df['senkou_span_b'].values[index:]

    color = '#98D4FD' #set colour to initially blue.
    if senkou_span_a[index] < senkou_span_b[index]:
        color = '#F7B0B6' #change colour to red.

    xs_blue, ys_blue, xs_red, ys_red = [], [], [], []

    for i in range(0, len(senkou_span_a)):
        if color == '#98D4FD' and senkou_span_b[i] > senkou_span_a[i] or i == len(senkou_span_a) - 1:
            line_a  = senkou_span_a[index:i+1]
            line_b = senkou_span_b[index:i+1]
            line_date = dates[index:i+1]
            index = i
            color = '#F7B0B6' #need to change colour to red.
            xs_blue.append(np.append(line_date, line_date[::-1]))
            ys_blue.append(np.append(line_a, line_b[::-1]))
        elif color == '#F7B0B6' and senkou_span_a[i] >= senkou_span_b[i] or i == len(senkou_span_a) - 1:
            line_a  = senkou_span_a[index:i+1]
            line_b = senkou_span_b[index:i+1]
            line_date = dates[index:i+1]
            index = i
            color = '#98D4FD' #need to change colour to blue.
            xs_red.append(np.append(line_date, line_date[::-1]))
            ys_red.append(np.append(line_a, line_b[::-1]))

    patch_renders = []
    patch_renders.append(p.patches(xs_blue, ys_blue, color='#98D4FD', line_color = '#98D4FD', line_alpha = 0, fill_alpha = 0.20))
    patch_renders.append(p.patches(xs_red, ys_red, color='#F7B0B6', line_color = '#F7B0B6', line_alpha = 0, fill_alpha = 0.20))

    return patch_renders


# In[68]:

#Make plots.
top_plot, top_renders = make_plot(sourceInc_top, sourceDec_top, source_top, df)
r7 = fill_area(top_plot, df)
top_plot.title.text = 'Bitcoin Chart'

bottom_plot, bottom_renders = make_plot(sourceInc_bottom, sourceDec_bottom, source_bottom, df_rip)
r8 = fill_area(bottom_plot, df_rip)
bottom_plot.title.text = 'Ripple Chart'


# In[69]:

def calc_returns(df_x, df_y):
    '''Returns data frame consisting of returns for 2 currencies.'''
    start_date = df_x['Date'].min()
    end_date = df_x['Date'].max()

    if df_y['Date'].min() > df_x['Date'].min(): # Dataset Y starts later so we start there.
        start_date = df_y['Date'].min()

    if df_y['Date'].max() < df_x['Date'].max(): # Dataset Y ends earlier so we end there.
        end_date = df_y['Date'].max()

    d = {'Date': [], 'x': [], 'y': []}
    for i in range(0, len(df_x)-1):
        if df_x.ix[i, 'Date'] > start_date and df_x.ix[i, 'Date'] < end_date:
            ret = (df_x.ix[i,'Close'] - df_x.ix[i+1, 'Close'])/df_x.ix[i+1, 'Close']
            d['x'].append(ret)
            d['Date'].append(df_x.ix[i, 'Date'])

    for i in range(0, len(df_y)-1):
        if df_y.ix[i, 'Date'] > start_date and df_y.ix[i, 'Date'] < end_date:
            ret = (df_y.ix[i,'Close'] - df_y.ix[i+1, 'Close'])/df_y.ix[i+1, 'Close']
            d['y'].append(ret)

    df_returns = pd.DataFrame(data = d)
    return df_returns


# In[70]:

'''Construct scatter correlation plot with market index.'''
df_y = df #Bitcoin as dependent variable.
df_x = df_market #Crix as independent variable.

df_returns_2 = calc_returns(df_x, df_y)
sourceCorr = ColumnDataSource(data = df_returns_2)
corr_plot = figure(plot_width=500, plot_height=350,
             tools='pan,wheel_zoom,box_select,reset', title = "Bitcoin vs. Crix Returns")
corr_plot.circle('x', 'y', size=2, source=sourceCorr,
            selection_color="#F7B0B6", alpha=0.6, nonselection_alpha=0.1, selection_alpha=0.4)


# In[71]:

def calc_returns_for_metrics(df):
    '''Calculate and return a list of returns.'''
    returns = []
    for i in range(0, len(df)-1):
        # Computes daily returns
        rp = (df.ix[i,'Close'] - df.ix[i+1, 'Close'])/df.ix[i+1, 'Close']
        returns.append(tuple((df.ix[i, 'Date'], rp)))
    #Sort returns by earliest to latest date.
    returns = sorted(returns, key=lambda L: L[0])
    return returns


# In[72]:

def get_data_between_dates(returns, start_date, end_date):
    '''Returns data between specified dates'''
    returns = [x for x in returns if x[0] >= datetime.strptime(start_date, "%Y-%m-%d")
               and x[0] <= datetime.strptime(end_date, "%Y-%m-%d")]
    returns = [x[1] for x in returns]
    return returns


# In[73]:

def calc_sharpe(df, rf, start_date, end_date):
    '''Calculate the annualised Sharpe Ratio using a risk free rate between a specified date range'''
    returns = calc_returns_for_metrics(df)
    returns = get_data_between_dates(returns, start_date, end_date)
    rp_mean = np.mean(returns)
    rp_sd = np.std(returns)
    return ((rp_mean-rf)/rp_sd) * (365/np.sqrt(365))


# In[74]:

def calc_sortino(df, rf, mar, start_date, end_date):
    '''Calculates the annualised Sortino ratio by using standard deviation of negative returns between a specified date range.'''
    rp = calc_returns_for_metrics(df)
    rp = get_data_between_dates(rp, start_date, end_date)
    neg_rp = [x for x in rp if x < mar]
    return ((np.mean(rp)) - rf)/np.std(neg_rp) * (365/np.sqrt(365))


# In[75]:

def calc_treynor(df, market, rf, start_date, end_date, beta):
    '''Calculates the Treynor Ratio between a specified date range.'''
    rp = calc_returns_for_metrics(df)
    rb = calc_returns_for_metrics(market)
    rp = get_data_between_dates(rp, start_date, end_date)
    rb = get_data_between_dates(rb, start_date, end_date)
    return ((np.mean(rp) - rf)/beta)


# In[76]:

def calc_infoRatio(df, market, start_date, end_date, alpha, beta):
    '''Calculates the Information Ratio between a specified date range.'''
    y = calc_returns_for_metrics(df)
    x = calc_returns_for_metrics(market)

    y = get_data_between_dates(y, start_date, end_date)
    x = get_data_between_dates(x, start_date, end_date)

    residuals = []
    for i in range(0, len(x)):
        predicted = alpha + beta*x[i]
        residuals = np.append(residuals, y[i] - predicted)

    return (alpha/np.std(residuals))


# In[77]:

def compute_regression(df):
    '''Computes the multiple metrics from regression of 2 datasets.'''
    df_x = df['x']
    df_y = df['y']
    beta, alpha, r_value, p_value, std_err = stats.linregress(df_x, df_y)
    r2 = r_value**2 # R-squared.
    metrics = [beta, alpha, r_value, r2, p_value, std_err]
    final_metrics = np.around(metrics, decimals=5) # To get 2 DP.
    return final_metrics


# In[78]:

'''Creates data-table of correlation.'''
metrics = compute_regression(df_returns_2)
START_DATE = '2016-11-22'
END_DATE = '2017-11-22'

#Calculate sharpe, sortino, treynor and info ratios.
sharpe = np.around(calc_sharpe(df_y,  0, START_DATE, END_DATE), decimals = 5)
sortino = np.around(calc_sortino(df_y, 0, 0, START_DATE, END_DATE), decimals = 5)
treynor = np.around(calc_treynor(df_y, df_x, 0, START_DATE, END_DATE, metrics[0]), decimals = 5)
info = np.around(calc_infoRatio(df_y, df_x, START_DATE, END_DATE, metrics[1], metrics[0]), decimals = 5)

reg_data = dict(metrics=["Beta", "P-value of Beta", "Alpha", "Correlation", "R-squared", "Sharpe", "Sortino", "Treynor", "Info"],
                values=[metrics[0], metrics[4], metrics[1], metrics[2], metrics[3], sharpe, sortino, treynor, info],
            )
reg_source = ColumnDataSource(reg_data)
columns_table = [
        TableColumn(field="metrics", title="Metrics"),
        TableColumn(field="values", title="Values"),
    ]
data_table = DataTable(source=reg_source, columns=columns_table, width=210, height=280, row_headers = False)


# In[79]:

def calc_summary_stats(df_x, df_y, calc_x = True, calc_y = True):
    '''Computes summary statistics for the returns of the x and y variable.'''
    df_returns = calc_returns(df_x, df_y)

    if calc_x == True:
        x_return_stats = np.around(df_returns['x'].describe(), decimals = 5)
        x_price_stats = np.around(df_x['Close'].describe(), decimals = 5)
        x_stats_data = dict(stats = ["Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
                            returns = [x_return_stats[1], x_return_stats[2], x_return_stats[3], x_return_stats[4],
                                      x_return_stats[5], x_return_stats[6], x_return_stats[7]],
                            prices = [x_price_stats[1], x_price_stats[2], x_price_stats[3], x_price_stats[4], x_price_stats[5],
                                      x_price_stats[6], x_price_stats[7]])
        if calc_y == False:
            return x_stats_data

    if calc_y == True:
        y_return_stats = np.around(df_returns['y'].describe(), decimals = 5)
        y_price_stats = np.around(df_y['Close'].describe(), decimals = 5)
        y_stats_data = dict(stats = ["Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
                            returns = [y_return_stats[1], y_return_stats[2], y_return_stats[3], y_return_stats[4],
                                          y_return_stats[5], y_return_stats[6], y_return_stats[7]],
                            prices = [y_price_stats[1], y_price_stats[2], y_price_stats[3], y_price_stats[4], y_price_stats[5],
                                          y_price_stats[6], y_price_stats[7]])
        if calc_x == False:
            return y_stats_data

    return x_stats_data, y_stats_data


# In[80]:

#Creating the summary statistics tables
x_sum_stats, y_sum_stats = calc_summary_stats(df_x, df_y)
x_table_source = ColumnDataSource(x_sum_stats)
columns_x_table = [
        TableColumn(field="stats", title="Statistic"),
        TableColumn(field="returns", title="Returns"),
        TableColumn(field="prices", title = "Prices"),
    ]
x_table = DataTable(source=x_table_source, columns=columns_x_table, width=210, height=280, row_headers = False)

y_table_source = ColumnDataSource(y_sum_stats)
columns_y_table = [
        TableColumn(field="stats", title="Statistic"),
        TableColumn(field="returns", title="Returns"),
        TableColumn(field="prices", title = "Prices"),
    ]
y_table = DataTable(source=y_table_source, columns=columns_y_table, width=210, height=280, row_headers = False)


# In[81]:

#Adding button widgets
button_1m = Button(label = "1 month", width = 80)
button_3m = Button(label= "3 month", width = 80)
button_6m = Button(label = "6 month", width = 80)
button_1y = Button(label = "1 Year", width = 80)
button_ytd = Button(label = "YTD", width = 80)
button_all = Button(label = "All", width = 80)


# In[82]:

START_OF_YEAR = "2017-1-01"

def update_1m():
    '''Update zoom to 1 month.'''
    df_1m = df.iloc[-30:,]['Date']
    top_plot.x_range.start = df_1m.min()
    top_plot.x_range.end = df_1m.max()
    bottom_plot.x_range.start = df_1m.min()
    bottom_plot.x_range.end = df_1m.max()


# In[83]:

def update_3m():
    '''Update zoom to 3 months.'''
    df_3m = df.iloc[-90:,]['Date']
    top_plot.x_range.start = df_3m.min()
    top_plot.x_range.end = df_3m.max()
    bottom_plot.x_range.start = df_3m.min()
    bottom_plot.x_range.end = df_3m.max()


# In[84]:

def update_6m():
    '''Update zoom to 6 months.'''
    df_6m = df.iloc[-180:,]['Date']
    top_plot.x_range.start = df_6m.min()
    top_plot.x_range.end = df_6m.max()
    bottom_plot.x_range.start = df_6m.min()
    bottom_plot.x_range.end = df_6m.max()


# In[85]:

def update_1y():
    '''Update zoom to 12 months.'''
    df_1y = df.iloc[-365:,]['Date']
    top_plot.x_range.start = df_1y.min()
    top_plot.x_range.end = df_1y.max()
    bottom_plot.x_range.start = df_1y.min()
    bottom_plot.x_range.end = df_1y.max()


# In[86]:

def update_ytd():
    '''Update zoom to start of year.'''
    top_plot.x_range.start = datetime.strptime(START_OF_YEAR, "%Y-%m-%d")
    top_plot.x_range.end = df['Date'].max()
    bottom_plot.x_range.start = datetime.strptime(START_OF_YEAR, "%Y-%m-%d")
    bottom_plot.x_range.end = df['Date'].max()


# In[87]:

def update_all():
    '''Update zoom to display all data.'''
    top_plot.x_range.start = df['Date'].min()
    top_plot.x_range.end = df['Date'].max()
    bottom_plot.x_range.start = df['Date'].min()
    bottom_plot.x_range.end = df['Date'].max()


# In[88]:

button_1m.on_click(update_1m)
button_3m.on_click(update_3m)
button_6m.on_click(update_6m)
button_1y.on_click(update_1y)
button_ytd.on_click(update_ytd)
button_all.on_click(update_all)


# In[89]:

#Adding checkbox button group widget
checkbox_button_group = CheckboxButtonGroup(labels=['Ichimoku', 'Volume'], active=[0, 1])

def update_renders(active):
    '''Toggle ichimoku and volume renders to make visible.'''
    #Make patches visible.
    r7[0].visible = 0 in checkbox_button_group.active
    r7[1].visible = 0 in checkbox_button_group.active
    r8[0].visible = 0 in checkbox_button_group.active
    r8[1].visible = 0 in checkbox_button_group.active
    #Make ichimoku renders visible.
    for i in range(0, len(top_renders) - 1):
        top_renders[i].visible = 0 in checkbox_button_group.active
        bottom_renders[i].visible = 0 in checkbox_button_group.active
    #Make volume renders visible.
    top_renders[-1].visible = 1 in checkbox_button_group.active
    bottom_renders[-1].visible = 1 in checkbox_button_group.active

checkbox_button_group.on_click(update_renders)


# In[90]:

# Create dropdown widgets.
DEFAULT_TICKERS = ['Bitcoin', 'Ripple', 'Ethereum', 'Crix']

def nix(val, lst):
    '''Remove currently selected currency from dropdown list.'''
    return [x for x in lst if x != val]

dropdown_top = Select(value = "Bitcoin", width = 215, options=nix('Crix', DEFAULT_TICKERS), title = 'Y variable:')
dropdown_bottom = Select(value = "Crix", width = 215, options=nix('Bitcoin', DEFAULT_TICKERS), title = 'X variable:')


# In[91]:

def update_metrics(df_x, df_y):
    '''Update source for metrics table'''
    # Update returns.
    new_returns = calc_returns(df_x, df_y)
    new_sourceCorr = new_returns
    sourceCorr.data.update(new_sourceCorr)

    # Update correlation metrics.
    new_reg_metrics = compute_regression(new_returns)
    new_sharpe = np.around(calc_sharpe(df_y,  0, START_DATE, END_DATE), decimals = 5)
    new_sortino = np.around(calc_sortino(df_y, 0, 0, START_DATE, END_DATE), decimals = 5)
    new_treynor = np.around(calc_treynor(df_y, df_x, 0, START_DATE, END_DATE, new_reg_metrics[0]), decimals = 5)
    new_info = np.around(calc_infoRatio(df_y, df_x, START_DATE, END_DATE, new_reg_metrics[1], new_reg_metrics[0]), decimals = 5)

    new_reg_data = dict(metrics=["Beta", "P-value of Beta", "Alpha", "Correlation",
                                 "R-squared", "Sharpe", "Sortino", "Treynor", "Info"],
                        values=[new_reg_metrics[0], new_reg_metrics[4], new_reg_metrics[1],
                                new_reg_metrics[2], new_reg_metrics[3], new_sharpe, new_sortino,
                                new_treynor, new_info],
                        )
    new_reg_sourceCorr = new_reg_data
    reg_source.data.update(new_reg_sourceCorr)


# In[92]:

def update_top_source(df, market = False):
    '''Update source for top plot and x data for correlation plot and tables.'''
    global df_x, df_y, r7

    if market == False:  # Don't create ichimoku plots for CRIX data.
        top_plot.renderers.remove(r7[0])
        top_plot.renderers.remove(r7[1])
        r7 = fill_area(top_plot, df)

        if 0 not in checkbox_button_group.active:
            r7[0].visible = False
            r7[1].visible = False

        inc = df.Close > df.Open
        dec = df.Open > df.Close
        newSourceInc = ColumnDataSource(ColumnDataSource.from_df(df.loc[inc]))
        newSourceDec = ColumnDataSource(ColumnDataSource.from_df(df.loc[dec]))
        newSource = ColumnDataSource(data = df)

        sourceInc_top.data.update(newSourceInc.data)
        sourceDec_top.data.update(newSourceDec.data)
        source_top.data.update(newSource.data)

        top_plot.title.text = '%s Chart' % (dropdown_top.value)

    df_y = df
    update_metrics(df_x, df_y)
    new_y_table_source = calc_summary_stats(df_x, df_y, False, True)
    y_table_source.data.update(new_y_table_source)
    corr_plot.title.text = '%s vs. %s Returns' % (dropdown_top.value, dropdown_bottom.value)


# In[95]:

def update_bottom_source(df, market = False):
    '''Update source for bottom plot and y data for correlation plot and table.'''
    global df_x, df_y, r8

    if market == False: #Don't create ichimoku plots for CRIX data.
        bottom_plot.renderers.remove(r8[0])
        bottom_plot.renderers.remove(r8[1])
        r8 = fill_area(bottom_plot, df)

        if 0 not in checkbox_button_group.active:
            r8[0].visible = False
            r8[1].visible = False

        inc = df.Close > df.Open
        dec = df.Open > df.Close
        newSourceInc = ColumnDataSource(ColumnDataSource.from_df(df.loc[inc]))
        newSourceDec = ColumnDataSource(ColumnDataSource.from_df(df.loc[dec]))
        newSource = ColumnDataSource(data = df)

        sourceInc_bottom.data.update(newSourceInc.data)
        sourceDec_bottom.data.update(newSourceDec.data)
        source_bottom.data.update(newSource.data)

        bottom_plot.title.text = '%s Chart' % (dropdown_bottom.value)

    df_x = df
    update_metrics(df_x, df_y)
    new_x_table_source = calc_summary_stats(df_x, df_y, True, False)
    x_table_source.data.update(new_x_table_source)
    corr_plot.title.text = '%s vs. %s Returns' % (dropdown_top.value, dropdown_bottom.value)


# In[96]:

def update_top_plot(attrname, old, new):
    '''Update top plot to selected data set'''
    dropdown_bottom.options = nix(new, DEFAULT_TICKERS)
    if dropdown_top.value == 'Crix':
        update_top_source(df_market, True)
    if dropdown_top.value == 'Ripple':
        update_top_source(df_rip)
    if dropdown_top.value == 'Bitcoin':
        update_top_source(df)
    if dropdown_top.value == 'Ethereum':
        update_top_source(df_eth)


# In[97]:

def update_bottom_plot(attrname, old, new):
    '''Update bottom plot to selected data set'''
    dropdown_top.options = nix(new, DEFAULT_TICKERS)
    if dropdown_bottom.value == 'Crix':
        update_bottom_source(df_market, True)
    if dropdown_bottom.value == 'Ripple':
        update_bottom_source(df_rip)
    if dropdown_bottom.value == 'Bitcoin':
        update_bottom_source(df)
    if dropdown_bottom.value == 'Ethereum':
        update_bottom_source(df_eth)

dropdown_top.on_change('value', update_top_plot)
dropdown_bottom.on_change('value', update_bottom_plot)


# In[98]:

#Format layout and display plot
button_controls = row([button_1m, button_3m, button_6m, button_1y, button_ytd, button_all, Spacer(width=50),
                       checkbox_button_group])
tables = row(y_table, Spacer(width = 20), column(x_table))
dropdown_and_tables = column(row(dropdown_top, Spacer(width = 15), dropdown_bottom, Spacer(width = 50)), tables)
top_row = row(corr_plot, column(Spacer(height = 25, width = 0), data_table), Spacer(width = 20), dropdown_and_tables)
price_plots = column(column(button_controls, top_plot, bottom_plot))

output_file("dashboard.html", title="dashboard.py")
layout = column(top_row, price_plots)
curdoc().add_root(layout)
show(layout)
