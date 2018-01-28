
# coding: utf-8

# In[83]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
from scipy import stats


# In[84]:

def ichimoku_plot(df):
    '''Computes the Ichimoku Kink? Hy? trend identification system.'''
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


# In[85]:

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


# In[86]:

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


# In[87]:

def scale_volume(dataframe, scale_factor): 
    '''Append a column of volumes scaled down by specified factor'''
    dataframe['Scaled Volume'] = dataframe['Volume']/scale_factor
    return dataframe 


# In[88]:

def sma_plot(df, window):
    '''Computes simple moving average.'''
    rolling = df['Close'].rolling(window=window) # Window tells us how many days average to take.
    return rolling.mean()


# In[89]:

def bollinger_plot(df, window, num_sd):
    '''Computes Bollinger bands depending on number of standard deviation and window.''' 
    rolling_mean = df['Close'].rolling(window).mean() # Window should be same as SMA.
    rolling_std = df['Close'].rolling(window).std()
    
    bollinger = pd.DataFrame(data=None)
    bollinger['Rolling Mean'] = rolling_mean
    bollinger['Bollinger High'] = rolling_mean + (rolling_std * num_sd)
    bollinger['Bollinger Low'] = rolling_mean - (rolling_std * num_sd)

    return bollinger


# In[90]:

import warnings
warnings.filterwarnings('ignore') # Warnings were getting annoying.

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


# In[91]:

from bokeh.events import ButtonClick
from bokeh.layouts import column, row, widgetbox
from bokeh.models.widgets import Button, Select
from bokeh.plotting import figure, output_file, show, ColumnDataSource, curdoc
from bokeh.models import HoverTool, CustomJS, Legend
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.io import output_file, show
from math import pi 


# In[92]:

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

def make_plot(sourceInc, sourceDec, source, df):
    '''Construct top and bottom candle + ichimoku plot'''
    w = 12*60*60*1000 # Half day in ms.

    # Display last 6 months by default.
    df_6m = df.iloc[-180:,]['Date']

    TOOLS = "pan, wheel_zoom, xbox_select, reset, save, hover"
    p = figure(x_axis_type = "datetime", tools = TOOLS,
               plot_width = 1000, plot_height = 400, x_range = (df_6m.min(), df_6m.max()), active_drag = "xbox_select") 
    p.xaxis.major_label_orientation = pi/4
    p.grid.grid_line_alpha = 0.30 

    # Construct increasing and decreasing lines.
    p.segment('Date', 'High', 'Date', 'Low', color="#17BECF", source = sourceInc)
    p.segment('Date', 'High', 'Date', 'Low', color="#FF7777", source = sourceDec)

    # Construct increasing and decreasing bars. 
    p.vbar('Date', w, 'Open', 'Close', fill_color="#17BECF", line_color="#17BECF", source = sourceInc)
    p.vbar('Date', w, 'Open', 'Close', fill_color="#FF7777", line_color="#FF7777", source = sourceDec)


    #Constructing hover tool for candle plot. Currently not working. TODO: fix this 
    #hover = p.select(dict(type=HoverTool))
    #hover.mode = "vline"
    '''hover.tooltips = [("Date", "@Date{%F}"),
                      ("Open", "@Open{0.2f}"), 
                      ("High", "@High{0.2f}"), 
                      ("Low", "@Low{0.2f}"), 
                      ("Close", "@Close{0.2f}")
                      ("Volume", "@Volume"]'''
    #hover.formatters = {'Date' : 'datetime'}

    # Add line render to display ichimoku plot.
    r1 = p.line('Date', 'tenkan_sen', line_width = 1, color = "#92FFB4", source = source)
    r2 = p.line('Date', 'kijun_sen', line_width = 1, color = "#92B2FF", source = source)
    r3 = p.line('Date', 'senkou_span_a', line_width = 1, color = "#C092FF", source = source)
    r4 = p.line('Date', 'senkou_span_b', line_width = 1, color = "#FFE592", source = source)
    r5 = p.line('Date', 'chikou_span', line_width = 1, color = "#6878FB", source = source)

    # Add bar render to display scaled down volume. TODO: make generalised scale for all currencies. 
    r6 = p.vbar('Date', w, 'Scaled Volume', 0, color="#5DE0F6", source = source)
    
    return p

def fill_area(p, df):  
    '''Fill area between senkou span A and B.'''
    # TODO: colour red if B above A'
    index = 0
    index_a = np.argwhere(np.isnan(df['senkou_span_a'].values)).max()
    index_b = np.argwhere(np.isnan(df['senkou_span_b'].values)).max()
    if index_b > index_a: 
        index = index_b
    else:
        index = index_a 
    dates = df['Date'].values[index:]
    senkou_span_a = df['senkou_span_a'].values[index:]
    senkou_span_b = df['senkou_span_b'].values[index:]
    band_y = np.append(senkou_span_a, senkou_span_b[::-1])
    band_x = np.append(dates, dates[::-1])
    return p.patch(band_x, band_y, color='#8EF3DA', fill_alpha = 0.20)

top_plot = make_plot(sourceInc_top, sourceDec_top, source_top, df)
r7 = fill_area(top_plot, df)
top_plot.title.text = 'Bitcoin Chart'

bottom_plot = make_plot(sourceInc_bottom, sourceDec_bottom, source_bottom, df_rip)
r8 = fill_area(bottom_plot, df_rip)
bottom_plot.title.text = 'Ripple Chart'

#Display legend. TODO: adjust legend to make visible 
'''legend = Legend(items = [
    ("Tenkan-Sen", [r1]),
    ("Kijun-Sen", [r2]),
    ("Senkou Span A", [r3]),
    ("Senkou Span B", [r4]),
    ("Chikou Span", [r5])
], location = (0,399))
top_plot.add_layout(legend, 'right')'''


# In[93]:

def calc_returns(df_x, df_y): 
    '''Return data frame consisting of returns for 2 currencies.'''
    start_date = df_x['Date'].min()
    end_date = df_x['Date'].max()
    
    if df_y['Date'].min() > df_x['Date'].min(): # Dataset Y starts later so we start there. 
        start_date = df_y['Date'].min()
    
    if df_y['Date'].max() < df_x['Date'].max(): # Dataset Y ends earlier so we end there. 
        end_date = df_y['Date'].max()

    d = {'x': [], 'y': []} 
    for i in range(0, len(df_x)-1): 
        if df_x.ix[i, 'Date'] > start_date and df_x.ix[i, 'Date'] < end_date: 
            ret = (df_x.ix[i+1,'Close'] - df_x.ix[i, 'Close'])/df_x.ix[i, 'Close']
            d['x'].append(ret)

    for i in range(0, len(df_y)-1): 
        if df_y.ix[i, 'Date'] > start_date and df_y.ix[i, 'Date'] < end_date:
            ret = (df_y.ix[i+1,'Close'] - df_y.ix[i, 'Close'])/df_y.ix[i, 'Close']
            d['y'].append(ret)

    df_returns = pd.DataFrame(data = d)
    return df_returns


# In[94]:

def compute_regression(df):
    '''Computes the multiple metrics from regression of 2 datasets.'''
    df_x = df['x']
    df_y = df['y']
    beta, alpha, r_value, p_value, std_err = stats.linregress(df_x, df_y)
    r2 = r_value**2 # R-squared.
    metrics = [beta, alpha, r_value, r2, p_value, std_err]
    return metrics


# In[95]:

'''Construct scatter correlation plot with market index.'''
df_x = df
df_y = df_market 

df_returns_2 = calc_returns(df_x, df_y)
sourceCorr = ColumnDataSource(data = df_returns_2)
corr_plot = figure(plot_width=400, plot_height=350,
             tools='pan,wheel_zoom,box_select,reset', title = "Bitcoin vs. Crix Returns")
corr_plot.circle('x', 'y', size=2, source=sourceCorr,
            selection_color="orange", alpha=0.6, nonselection_alpha=0.1, selection_alpha=0.4)


# In[100]:

'''Creates data-table of correlation.'''
metrics = compute_regression(df_returns_2)
reg_data = dict(metrics=["Beta", "P-value of Beta", "Alpha", "Correlation", "R-squared"],
                values=[metrics[0], metrics[4], metrics[1], metrics[2], metrics[3]],
            )
reg_source = ColumnDataSource(reg_data)
columns_table = [
        TableColumn(field="metrics", title="Metrics"),
        TableColumn(field="values", title="Values"),
    ]
data_table = DataTable(source=reg_source, columns=columns_table, width=400, height=280)


# In[97]:

#Adding button widgets
button_3m = Button(label= "3 month", width = 80)
button_6m = Button(label = "6 month", width = 80)
button_1y = Button(label = "1 Year", width = 80)
button_ytd = Button(label = "YTD", width = 80)
button_all = Button(label = "All", width = 80)

def update_3m(): 
    '''Update zoom to 3 months'''
    df_3m = df.iloc[-90:,]['Date']
    top_plot.x_range.start = df_3m.min()
    top_plot.x_range.end = df_3m.max()
    bottom_plot.x_range.start = df_3m.min()
    bottom_plot.x_range.end = df_3m.max()

def update_6m():
    '''Update zoom to 6 months'''
    df_6m = df.iloc[-180:,]['Date']
    top_plot.x_range.start = df_6m.min()
    top_plot.x_range.end = df_6m.max()
    bottom_plot.x_range.start = df_6m.min()
    bottom_plot.x_range.end = df_6m.max()

def update_1y():
    '''Update zoom to 12 months'''
    df_1y = df.iloc[-365:,]['Date']
    top_plot.x_range.start = df_1y.min()
    top_plot.x_range.end = df_1y.max()
    bottom_plot.x_range.start = df_1y.min()
    bottom_plot.x_range.end = df_1y.max()

def update_ytd():
    '''Update zoom to start of year'''
    start_date = datetime.strptime("2017-1-01", "%Y-%m-%d")
    top_plot.x_range.start = start_date
    top_plot.x_range.end = df['Date'].max()
    bottom_plot.x_range.start = start_date
    bottom_plot.x_range.end = df['Date'].max()

def update_all(): 
    '''Update zoom to display all data'''
    top_plot.x_range.start = df['Date'].min() 
    top_plot.x_range.end = df['Date'].max()
    bottom_plot.x_range.start = df['Date'].min()
    bottom_plot.x_range.end = df['Date'].max()
    
button_3m.on_click(update_3m)
button_6m.on_click(update_6m)
button_1y.on_click(update_1y)
button_ytd.on_click(update_ytd)
button_all.on_click(update_all)


# In[98]:

# Create dropdown widgets. 
DEFAULT_TICKERS = ['Bitcoin', 'Ripple', 'Ethereum', 'Crix']

def nix(val, lst):
    '''Remove currently selected currency from dropdown list.'''
    return [x for x in lst if x != val]

dropdown_top = Select(value = "Bitcoin", options=nix('Crix', DEFAULT_TICKERS))
dropdown_bottom = Select(value = "Crix", options=nix('Bitcoin', DEFAULT_TICKERS))

def update_top_source(df, market = False): 
    '''Update source for top plot and x data for correlation plot.'''
    global df_x, df_y, r7 

    if market == False:  # Don't create ichimoku plots for CRIX data.
        top_plot.renderers.remove(r7)
        r7 = fill_area(top_plot, df)

        inc = df.Close > df.Open
        dec = df.Open > df.Close
        newSourceInc = ColumnDataSource(ColumnDataSource.from_df(df.loc[inc]))
        newSourceDec = ColumnDataSource(ColumnDataSource.from_df(df.loc[dec]))
        newSource = ColumnDataSource(data = df)

        sourceInc_top.data.update(newSourceInc.data)
        sourceDec_top.data.update(newSourceDec.data)
        source_top.data.update(newSource.data)
        
        top_plot.title.text = '%s Chart' % (dropdown_top.value)

    df_x = df 
    # Update returns.
    new_returns = calc_returns(df_x, df_y) 
    new_sourceCorr = new_returns 
    sourceCorr.data.update(new_sourceCorr)
    
    # Update correlation metrics.
    new_reg_metrics = compute_regression(new_returns)
    new_reg_data = dict(metrics=["Beta", "P-value of Beta", "Alpha", "Correlation", "R-squared"],
                values=[new_reg_metrics[0], new_reg_metrics[4], new_reg_metrics[1], 
                        new_reg_metrics[2], new_reg_metrics[3]],
            )
    new_reg_sourceCorr = new_reg_data
    reg_source.data.update(new_reg_sourceCorr) 
    
    corr_plot.title.text = '%s vs. %s Returns' % (dropdown_top.value, dropdown_bottom.value)

def update_bottom_source(df, market = False):
    '''Update source for bottom plot and y data for correlation plot.'''
    global df_x, df_y, r8

    if market == False: #Don't create ichimoku plots for CRIX data.
        bottom_plot.renderers.remove(r8)
        r8 = fill_area(bottom_plot, df)

        inc = df.Close > df.Open
        dec = df.Open > df.Close
        newSourceInc = ColumnDataSource(ColumnDataSource.from_df(df.loc[inc]))
        newSourceDec = ColumnDataSource(ColumnDataSource.from_df(df.loc[dec]))
        newSource = ColumnDataSource(data = df)

        sourceInc_bottom.data.update(newSourceInc.data)
        sourceDec_bottom.data.update(newSourceDec.data)
        source_bottom.data.update(newSource.data)

        bottom_plot.title.text = '%s Chart' % (dropdown_bottom.value)

    df_y = df
    # Update returns.
    new_returns = calc_returns(df_x, df_y)
    new_sourceCorr = new_returns 
    sourceCorr.data.update(new_sourceCorr)
    # Update correlation metrics.
    new_reg_metrics = compute_regression(new_returns)
    new_reg_data = dict(metrics=["Beta", "P-value of Beta", "Alpha", "Correlation", "R-squared"],
                values=[new_reg_metrics[0], new_reg_metrics[4], new_reg_metrics[1], 
                        new_reg_metrics[2], new_reg_metrics[3]],
            )
    new_reg_sourceCorr = new_reg_data
    reg_source.data.update(new_reg_sourceCorr) 
    
    corr_plot.title.text = '%s vs. %s Returns' % (dropdown_top.value, dropdown_bottom.value)
    
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


# In[99]:

#Format layout and display plot
button_controls = row([button_3m, button_6m, button_1y, button_ytd, button_all])
dropdown_controls = row(corr_plot, column(dropdown_top, dropdown_bottom))
price_plots = column(column(button_controls, top_plot), bottom_plot)

output_file("dashboard.html", title="dashboard.py")
layout = column(dropdown_controls, widgetbox(data_table), price_plots)
curdoc().add_root(layout)
show(layout)


# In[ ]:



