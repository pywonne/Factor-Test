'''
CNE6 Factor Test
@author Yiwen Pan
'''
from higgsboom.MarketData.CSecurityMarketDataUtils import *
secUtils = CSecurityMarketDataUtils('Z:/StockData')
from higgsboom.FuncUtils.DateTime import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
import statsmodels.api as sm
import pickle 

# 获取股票市值大小
LNCAP_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_LNCAP.pkl','rb')
LNCAP = pd.DataFrame(pickle.load(LNCAP_file)).T # index是股票，columns是日期
CAP = np.exp(LNCAP)

# 市值因子
LIQUIDITY_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_Liquidity_sum.pkl','rb')
SIZE = pd.DataFrame(pickle.load(LIQUIDITY_file)) # index是股票，columns是日期
SIZE = SIZE.loc['2019-01-07':'2021-05-14',:].T


def Winsorize(x,nsigma=5):
    '''
    去极值
    '''
    md = np.nanmedian(x)    
    mad = np.nanmedian(np.abs(x - md))
    mad_e = 1.483 * mad
    x[x > md + nsigma*mad_e] = md + nsigma*mad_e
    x[x < md - nsigma*mad_e] = md - nsigma*mad_e
    return x

def CAP_Standard(array, floatcap):
    '''
    风格因子通过流通市值标准化
    '''
    x = array.replace([np.inf, -np.inf], np.nan).dropna()
    y = floatcap.replace([np.inf, -np.inf], np.nan).dropna()
    intersectID = list(set(x.index) & set(y.index))
    intersectID.sort()
    x = x[intersectID]
    y = y[intersectID]
    average = np.average(x, weights=y)
    std = np.sqrt(np.average((x-average)**2, weights=y))
    temp = x - average 
    return temp / std if std !=0 else temp

def Ind_Cap_Neutralize(data, ind_data, cap_data):
    '''
    行业市值中性化
    因子序列对流动市值与行业哑变量做线性回归，取残差作为新的因子值
    '''
    ind_data['LNCAP'] = np.log(cap_data)   
    data_adj = pd.concat([data, ind_data], axis=1)

    data_adj.dropna(axis=0, how='any', inplace=True)
    
    
    for i in data_adj.columns:
        x = np.hstack((np.ones((len(data_adj),1)), data_adj.iloc[:, 1:].values))
        y = data_adj.loc[:,i].values
        model = sm.WLS(y, x, weights=np.sqrt(np.exp(data_adj['LNCAP'])))
        result = model.fit()
        res = result.resid
        data_adj.loc[:,i] = res
    ind_data = ind_data.drop(columns=['LNCAP'])
    return data_adj['SIZE']

def ret_regress(factor_exposure, factor_return):
    x = factor_exposure.values
    y = factor_return.values
    model = sm.OLS(y, x)
    result = model.fit()
    return result.tvalues


def daily_calculate(t):
    IndexCons = secUtils.IndexConstituents('000905.SH', t) # 中证500股票池
    Index500_SIZE = pd.Series(index=IndexCons, dtype=float)
    Index500_CAP = pd.Series(index=IndexCons, dtype=float)
    # print(IndexCons)
    for j in SIZE.index:
        if j in IndexCons:
            Index500_SIZE[j] = SIZE.at[j,t]
            Index500_CAP[j] = CAP.at[j,t]
        
    StyleFactors = pd.concat([Index500_SIZE], axis=1)
    StyleFactors.columns = ['SIZE']
    StyleFactors.dropna(axis=0, how='any', inplace=True)
    # print(StyleFactors)

    # 行业因子
    try:
        IndusCons = secUtils.DailyIndustryConstituents('SW', 'L1', t)   # 申万 L1, 28个行业分类
    except:
        IndusCons = secUtils.LatestIndustryConstituents('SW', 'L1')
    IndusList = []
    stock = []
    for i in StyleFactors.index:
        for key in IndusCons:
            if i in IndusCons.get(key):
                IndusList.append(key)
                stock.append(i)
                break
    IndusTmp = pd.DataFrame({'行业': IndusList})
    IndusFactor = pd.get_dummies(IndusTmp)
    diff = StyleFactors.index.difference(stock).values
    StyleFactors.drop(index=diff, inplace=True)
    Index500_CAP = Index500_CAP[StyleFactors.index]
    Index500_CAPWEIGHT = Index500_CAP / Index500_CAP.sum()
    IndusFactor.index = StyleFactors.index
    

    # 风格因子
    StyleFactors['SIZE'] = Winsorize(StyleFactors['SIZE'], 5)
    StyleFactors['SIZE'] = CAP_Standard(StyleFactors['SIZE'], Index500_CAPWEIGHT)
    StyleFactors['SIZE'] = Ind_Cap_Neutralize(StyleFactors['SIZE'], IndusFactor, Index500_CAP)
    
    # print(StyleFactors)
    nextDay = NextTradingDate(t)
    stock_forward_return = pd.Series(index=StyleFactors.index, dtype=float)
    for stock in StyleFactors.index:
        stockDailyFrame = secUtils.StockDailyDataFrame(stock, beginDt=t, endDt=nextDay)
        stockDailyFrame.index = stockDailyFrame['TRADING_DATE']
        if nextDay not in stockDailyFrame.index:
            StyleFactors.drop(index=[stock], inplace=True)
            stock_forward_return.drop(index=[stock], inplace=True)
            continue
        if(stockDailyFrame.at[nextDay, 'TRADE_STATUS'] != '交易') or (stockDailyFrame.shift(1).at[nextDay, 'TRADE_STATUS'] != '交易'):
            StyleFactors.drop(index=[stock], inplace=True)
            stock_forward_return.drop(index=[stock], inplace=True)
            continue 
        today_ret = stockDailyFrame.at[nextDay, 'CLOSE']
        yesterday_ret = stockDailyFrame.shift(1).at[nextDay, 'CLOSE']
        stock_forward_return[stock] = (today_ret / yesterday_ret) - 1
    # print(stock_forward_return)
    
    # tvalue = ret_regress(StyleFactors['SIZE'], stock_forward_return)
    # tvalue = np.abs(tvalue)
    # print(t, tvalue)

    ICvalue = StyleFactors['SIZE'].corr(stock_forward_return)
    print(t, ICvalue)
    return ICvalue


if __name__ == "__main__":
    PeriodList = TradingDays(startDate='2019-01-07', endDate='2021-05-14')
    pool = ThreadPool()
    value_series = pool.map(daily_calculate, PeriodList)
    mean = np.mean(value_series)
    std = np.std(value_series)
    print(mean, std)
    
   

# daily_f = daily_calculate('2019-08-08') 