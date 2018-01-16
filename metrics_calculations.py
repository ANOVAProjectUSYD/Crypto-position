import numpy as np
import pandas as pd


def calc_returns(df):
    '''Calculate and return a list of returns.'''
    returns = []
    for i in range(0, len(df)-1):
        # Computes daily returns
        rp = (df.ix[i, 'Close'] - df.ix[i+1, 'Close'])/(df.ix[i+1, 'Close'])
        returns = np.append(returns, rp)
    return returns

def calc_sharpe(df, rf):
    '''Calculate the Sharpe Ratio using a risk free rate.'''
    returns = calc_returns(df)
    rp_mean = np.mean(returns)
    rp_sd = np.std(returns)
    return ((rp_mean-rf)/rp_sd) * (365/np.sqrt(365))


def calc_sortino(df, rf, mar):
    '''Calculates the Sortino ratio by using standard deviation of negative returns.'''
    rp = calc_returns(df)
    neg_rp = [x for x in rp if x < mar]
    return ((np.mean(rp)) - rf)/np.std(neg_rp) * (365/np.sqrt(365))


def calc_treynor(df, rf, market):
    '''Calculates the Treynor Ratio,'''
    rp = calc_returns(df)
    rb = calc_returns(market)
    beta = np.cov(rp, rb)[0][1]/np.var(rb)
    return ((np.mean(rp) - rf)/beta) * (365/np.sqrt(365))


def calc_infoRatio(df, market):
    '''Calculates the Information Ratio.'''
    x = calc_returns(df)
    y = calc_returns(market)

    coef = np.polyfit(x, y, 1)
    beta = coef[0]
    alpha = coef[1]
    residuals = []

    for i in range(0, len(x)):
        predicted = alpha + beta*x[i]
        residuals = np.append(residuals, y[i] - predicted)

    residual_sd = np.std(residuals)
    return (alpha/residual_sd) * (365/np.sqrt(365))
