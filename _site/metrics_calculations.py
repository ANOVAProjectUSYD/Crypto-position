import numpy as np
import pandas as pd

############################################################
############ Computes financial metrics for currency. ######
############################################################


def calc_index_returns(df): 
    '''Calculate and return a list of benchmark returns'''
    index_returns = []
    for i in range(0, len(df)-1):
        #Compute daily returns 
        rb = (df.ix[i+1,'price'] - df.ix[i,'price'])/(df.ix[i,'price'])
        index_returns.append(tuple((df.ix[i+1, 'date'], rb)))
    #Sort returns by earliest to latest date 
    index_returns = sorted(index_returns, key=lambda L: datetime.strptime(L[0], '%d/%m/%Y'))
    return index_returns 

def calc_returns(df):
    '''Calculate and return a list of returns.'''
    returns = []
    for i in range(0, len(df)-1):
        # Computes daily returns
        rp = (df.ix[i, 'Close'] - df.ix[i+1, 'Close'])/(df.ix[i+1, 'Close'])
        returns.append(tuple((df.ix[i, 'Date'], rp)))
    #Sort returns by earliest to latest date 
    returns = sorted(returns, key=lambda L: datetime.strptime(L[0], '%d/%m/%Y'))
    return returns

def get_data_between_dates(returns, start_date, end_date):
    '''Returns data between specified dates'''
    returns = [x for x in returns if datetime.strptime(x[0],'%d/%m/%Y') 
               >= datetime.strptime(start_date, '%d/%m/%Y')
               and datetime.strptime(x[0],'%d/%m/%Y') <= datetime.strptime(end_date, '%d/%m/%Y')]
    returns = [x[1] for x in returns]
    return returns 

def calc_sharpe(df, rf, start_date, end_date):
    '''Calculate the annualised Sharpe Ratio using a risk free rate between a specified date range'''
    returns = calc_returns(df)
    returns = get_data_between_dates(returns, start_date, end_date)
    rp_mean = np.mean(returns)
    rp_sd = np.std(returns)
    return ((rp_mean-rf)/rp_sd) * (365/np.sqrt(365))

def calc_sortino(df, rf, mar, start_date, end_date):
    '''Calculates the annualised Sortino ratio by using standard deviation of negative returns between a specified date range.'''
    rp = calc_returns(df)
    rp = get_data_between_dates(rp, start_date, end_date)
    neg_rp = [x for x in rp if x < mar]
    return ((np.mean(rp)) - rf)/np.std(neg_rp) * (365/np.sqrt(365))

def calc_treynor(df, market, rf, start_date, end_date):
    '''Calculates the Treynor Ratio between a specified date range.'''
    rp = calc_returns(df)
    rb = calc_index_returns(market)
    rp = get_data_between_dates(rp, start_date, end_date)
    rb = get_data_between_dates(rb, start_date, end_date)
    beta = np.cov(rp, rb, ddof = 1)[0][1]/np.var(rb, ddof = 1)
    return ((np.mean(rp) - rf)/beta) 


def calc_infoRatio(df, market, start_date, end_date):
    '''Calculates the Information Ratio between a specified date range.'''
    y = calc_returns(df)
    x = calc_index_returns(market)
    
    y = get_data_between_dates(y, start_date, end_date)
    x = get_data_between_dates(x, start_date, end_date)
    
    coef = np.polyfit(x, y, 1)
    beta = coef[0]
    alpha = coef[1]
    residuals = []
    for i in range(0, len(x)):
        predicted = alpha + beta*x[i]
        residuals = np.append(residuals, y[i] - predicted)

    return (alpha/np.std(residuals)) 

# print('Sharpe ', calc_sharpe(bitcoin_df, 0, '22/11/2016', '22/11/2017'))
# print('Sortino ', calc_sortino(bitcoin_df, 0, 0, '22/11/2016', '22/11/2017'))
# print('Treynor ', calc_treynor(bitcoin_df, index_df, 0, '22/11/2016', '22/11/2017'))
# print('Info ', calc_infoRatio(bitcoin_df, index_df, '22/11/2016', '22/11/2017'))