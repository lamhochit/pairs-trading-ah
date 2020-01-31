import numpy as np
import pandas as pd

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

import matplotlib.pyplot as plt


# Load csv data into DataFrame, set column one as index
def data_loader(csv_file):
    try:
        data = pd.read_csv('/Users/lamhochit/PycharmProjects/project-alpha/pairs-data/' + csv_file, index_col=0)
        print('File ' + csv_file + ' has been loaded.')
        return data
    except FileNotFoundError:
        print('File ' + csv_file + ' not found.')


# Plots the data of SH and HK close
def data_plotter(df):
    data1 = df['HK_close']
    data2 = df['SH_close']

    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel='Price', xlabel='Date')
    data1.plot(ax=ax1)
    data2.plot(ax=ax1)

    plt.legend()
    plt.show()


# Check whether the Series is stationary (Input series only, doesn't work for DataFrame)
def stationary_check(sr, cutoff=0.05):
    pvalue = adfuller(sr)[1]
    if pvalue < cutoff:
        print('p-value = ' + str(pvalue) + ' The series ' + sr.name + ' is likely stationary.')
        return True
    else:
        print('p-value = ' + str(pvalue) + ' The series ' + sr.name + ' is likely non-stationary.')
        return False


# Get the beta for Z = sr2 - beta * sr1
def beta_regression(sr1, sr2):
    sr1 = sm.add_constant(sr1)
    results = sm.OLS(sr2, sr1).fit()
    print(results.params)
    return results.params['HK_close']


# Main Code
data = data_loader('0.csv')
stationary_check(data['HK_close'])
sr1 = data['HK_close']
sr2 = data['SH_close']


b = beta_regression(sr1, sr2)
z = sr2 - b * sr1
z.name = 'z'


plt.plot(z.index, z.values)
plt.xlabel('Time')
plt.ylabel('Series Value')
plt.legend([z.name])
stationary_check(z)
plt.show()


