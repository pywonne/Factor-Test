from higgsboom.MarketData.CSecurityMarketDataUtils import *
secUtils = CSecurityMarketDataUtils('Z:/StockData')
from higgsboom.FuncUtils.DateTime import *
from threading import Thread
import numpy as np
import pandas as pd
import pickle

# 获取股票列表
LNCAP_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_LNCAP.pkl', 'rb') 
CNE6_LNCAP = pd.DataFrame(pickle.load(LNCAP_file))
stockList = CNE6_LNCAP.columns
IndexList = range(len(stockList))

# Size 因子
# LNCAP_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_LNCAP.pkl', 'rb') 
# CNE6_LNCAP = pd.DataFrame(pickle.load(LNCAP_file))
# CNE6_LNCAP = CNE6_LNCAP[CNE6_LNCAP.index >= '2019-01-07']
# CNE6_LNCAP.to_pickle('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_LNCAP.pkl')
# MIDCAP_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_NLSIZE.pkl', 'rb')
# CNE6_MIDCAP = pd.DataFrame(pickle.load(MIDCAP_file))
# CNE6_MIDCAP = CNE6_MIDCAP[CNE6_MIDCAP.index >= '2019-01-07']
# CNE6_MIDCAP.to_pickle('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_MIDCAP.pkl')
# Size = (CNE6_LNCAP + CNE6_MIDCAP) / 2
# Size.to_pickle('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_SIZE.pkl')

# Volatility 因子
# BETA_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_BETA.pkl', 'rb') 
# CNE6_BETA = pd.DataFrame(pickle.load(BETA_file))
# CNE6_BETA = CNE6_BETA[CNE6_BETA.index >= '2019-01-07']
# CNE6_BETA.to_pickle('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_BETA.pkl')
# RESVOL_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_Volatility.pkl', 'rb')
# CNE6_RESVOL = pd.DataFrame(pickle.load(RESVOL_file))
# CNE6_RESVOL = CNE6_RESVOL[CNE6_RESVOL.index >= '2019-01-07']
# CNE6_RESVOL.to_pickle('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_RESIDUALVOL.pkl')
# Volatility = (CNE6_BETA + CNE6_RESVOL) / 2
# Volatility.to_pickle('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_VOLATILITY.pkl')

PeriodList_ATVR = TradingDays(startDate='2017-12-24', endDate='2021-05-14')

# Liquidity 因子
def get_stock_ATVR(stock, tau=63, length=252):
    print(stock)
    lambd = 0.5**(1./tau) 
    w = lambd ** np.arange(length)[::-1] # 指数权重
    stock_daily_df = secUtils.StockDailyDataFrame(stock, beginDt='2017-12-24', endDt='2021-05-14')
    t_series = pd.Series(dtype='float64')
    for t in range(0, len(PeriodList_ATVR)-251, 1):
        i_daily_df = stock_daily_df.iloc[t:t+252,:]
        if len(i_daily_df) == 252:
            Vt_st = i_daily_df['VOLUME'] / i_daily_df['SHARE_TOTALA'] 
            t_series[PeriodList_ATVR[t+251]] = np.sum(w * Vt_st.values)
        else:
            t_series[PeriodList_ATVR[t+251]] = np.nan
    t_series = t_series.to_frame()
    t_series.columns = [stock]
    print(t_series)
    t_series.to_excel(os.path.join('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_ATVR',stock+'.xlsx'))
    return t_series


# threads = []
# for val in IndexList:
#     threads.append(Thread(target=get_stock_ATVR, args=(stockList[val],)))
#     threads[val].start()
# for thread in threads:
#     thread.join()


PeriodList_STREV = TradingDays(startDate='2018-12-04', endDate='2021-05-14')

# Momentum 因子
def get_stock_STREV(stock, tau=5, length=21):
    print(stock)
    lambd = 0.5**(1./tau) 
    w = lambd ** np.arange(length)[::-1] # 指数权重
    stock_daily_df = secUtils.StockDailyDataFrame(stock, beginDt='2018-12-04', endDt='2021-05-14')
    t_series = pd.Series(dtype='float64')
    for t in range(1, len(PeriodList_STREV)-21, 1):
        i_yesterday_df = stock_daily_df.iloc[t-1:t+20,:].reset_index()
        i_today_df = stock_daily_df.iloc[t:t+21,:].reset_index()
        if len(i_yesterday_df) == 21 and len(i_today_df) == 21:
            i_return = (i_today_df['CLOSE'] / i_yesterday_df['CLOSE']) 
            t_series[PeriodList_STREV[t+21]] = np.sum(w * np.log(i_return.values))
        else:
            t_series[PeriodList_STREV[t+21]] = np.nan
    t_series = t_series.to_frame()
    t_series.columns = [stock]
    print(t_series)
    t_series.to_pickle(os.path.join('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_STREV',stock+'.pkl'))
    return t_series

# threads = []
# for val in IndexList:
#     threads.append(Thread(target=get_stock_STREV, args=(stockList[val],)))
#     threads[val].start()
# for thread in threads:
#     thread.join()

PeriodList_IndMom =  TradingDays(startDate='2018-07-03', endDate='2021-05-14')

def get_IndMom_stock(stock, tau=21, length=126):
    '''
    个股相对强度定义
    '''
    print(stock)
    lambd = 0.5**(1./tau) 
    w = lambd ** np.arange(length)[::-1] # 指数权重
    stock_daily_df = secUtils.StockDailyDataFrame(stock, beginDt='2018-07-03', endDt='2021-05-14')
    t_series = pd.Series(dtype='float64')
    for t in range(1, len(PeriodList_IndMom)-125, 1):
        i_yesterday_df = stock_daily_df.iloc[t-1:t+125,:].reset_index()
        i_today_df = stock_daily_df.iloc[t:t+126,:].reset_index()
        if len(i_yesterday_df) == 126 and len(i_today_df) == 126:
            i_return = i_today_df['CLOSE'] / i_yesterday_df['CLOSE']
            t_series[PeriodList_IndMom[t+125]] = np.sum(w * np.log(i_return.values))
        else:
            t_series[PeriodList_IndMom[t+125]] = np.nan
    t_series = t_series.to_frame()
    t_series.columns = [stock]
    print(t_series)





# Liquidity 因子
STOM_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_STOM.pkl', 'rb')
CNE6_STOM = pd.DataFrame(pickle.load(STOM_file))
print(CNE6_STOM)
# CNE6_STOM.to_pickle('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_STOM.pkl')

STOQ_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_STOQ.pkl', 'rb')
CNE6_STOQ = pd.DataFrame(pickle.load(STOQ_file))
print(CNE6_STOQ)
# CNE6_STOQ.to_pickle('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_STOQ.pkl')

STOA_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_STOA.pkl', 'rb')
CNE6_STOA = pd.DataFrame(pickle.load(STOA_file))
print(CNE6_STOA)
# CNE6_STOA.to_pickle('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_STOA.pkl')

LNCAP_file = open('C:/Users/panyi/Documents/BarraFactorsLibrary/Barra/CNE5_LNCAP.pkl', 'rb') 
CNE6_LNCAP = pd.DataFrame(pickle.load(LNCAP_file))
stockList = CNE6_LNCAP.columns
# CNE6_ATVR = pd.DataFrame()
# for stock in stockList:
#     i = pd.read_excel(os.path.join('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_ATVR',stock+'.xlsx'), header=0, index_col=0)
#     CNE6_ATVR = pd.concat([CNE6_ATVR, i], axis=1)
# print(CNE6_ATVR)
# CNE6_ATVR.to_pickle('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_ATVR.pkl')


# CNE6_Liquidity = (CNE6_STOM + CNE6_STOQ + CNE6_STOA + CNE6_ATVR) / 4
# print(CNE6_Liquidity)
# CNE6_Liquidity.to_pickle('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_LIQUIDITY.pkl')

# stockList = secUtils.AStockList('2021-05-14')
CNE6_STREV = pd.DataFrame()
for stock in stockList:
    i = open(os.path.join('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_STREV',stock+'.pkl'),'rb')
    i_data = pd.DataFrame(pickle.load(i))
    CNE6_STREV = pd.concat([CNE6_STREV, i_data], axis=1)
print(CNE6_STREV)
CNE6_STREV.to_pickle('C:/Users/panyi/Documents/BarraFactorsLibrary/CNE6/CNE6_STREV.pkl')

