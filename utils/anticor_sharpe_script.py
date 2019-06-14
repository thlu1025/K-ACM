# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:10:56 2019

@author: lth
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:22:49 2019

@author: lth
"""

from Class.KFfilter import KFfilter
from Class.anticor import Anticor
import numpy as np 
import pandas as pd





def max_drawdown(vec):
    drawdown = 0.
    max_seen = vec[0]
    for val in vec[1:]:
        max_seen = max(max_seen, val)
        drawdown = max(drawdown, 1 - val / max_seen)
    return drawdown


def anticor_algo_fixedAR(sharp_dat, dat, train_size, window, holding, init_weight, cost):
    '''
    Calculate the return, sharp, draw down for specific parameters:
    
    Parameters:
    ----
    sharp_dat  :pd.DataFrame
        the sharp_ratio data 
    dat        :pd.DataFrame
        price dat
    train_size :int
        size for training the filter
    window     :int
        window size for observing the anticorrelation
    holding    :int
        holding period for assets
    init_weight:np.array
        initial weight for the assets     
    cost       :float
        cost rate for each trade
    
    Return:
    ----
        return series
        sharpe ratio
        drawdown
    '''
    
    dim = dat.shape[1]
    #calculate filtered result
    filter_res = KFfilter.fixed_filter(sharp_dat, train_size, np.eye(dim), np.eye(dim))
    #denoise price
    CAPR_df = pd.concat([filter_res.iloc[:,i+dim]/filter_res.iloc[:,i]
                                                  for i in range(dim)],axis =1)
    CAPR_df.columns = dat.columns
    
    price_df = filter_res.iloc[:, dim:]
    
    #apply anticorrelation 
    anticor = Anticor(window, holding, CAPR_df, price_df, init_weight)
    anticor.getweight('both')
    #calculate return
    ret = anticor.calret(dat, cost)
    
    sharpe = ret.mean()/ret.std()*np.sqrt(24*365)
    
    down = max_drawdown(np.cumprod(1+ret))
    
    return ret, sharpe, down, anticor.weight_lst



def anticor_algo_dynamicAR(sharp_dat, dat, train_size, train_window, predict_size,
                           window, holding, init_weight, cost):
    '''
    Calculate the return, sharp, draw down for specific parameters using dynamic
    AR estimation:
        
    Parameters:
    ----
    sharp_dat   :pd.DataFrame
        sharp ratio data 
    dat         :pd.DataFrame
        price dat
    train_size  :int
        size for training the filter
    train_window:int
        size for training the AR coefficients
    predict_size:int
        size for using the AR coefficients
    window      :int
        window size for observing the anticorrelation
    holding     :int
        holding period for assets
    init_weight :np.array
        initial weight for the assets     
    cost        :float
        cost rate for each trade
    
    Return:
    ----
    pd.Series:
        return series
    float:    
        sharpe ratio
    float:    
        drawdown
    pd.DataFrame:
        weight_lst of the strategy
    '''
    
    dim = dat.shape[1]
    #calculate filtered result
    KF = KFfilter(sharp_dat, train_size, train_window, predict_size)
    KF.trainParam()
    
    obs_mat = np.eye(dim)
    filter_res = KF.adapt_filter(obs_mat)

    #denoise price
    #measured sharpe/ filtered sharpe
    CAPR_df = pd.concat([filter_res.iloc[:,i+dim]/filter_res.iloc[:,i] 
                                                  for i in range(dim)],axis =1)
    CAPR_df.columns = dat.columns
    
    price_df = filter_res.iloc[:, dim:]
    #apply anticorrelation 
    anticor = Anticor(window, holding, CAPR_df, price_df, init_weight)
    anticor.getweight('both')
    #calculate return
    ret = anticor.calret(dat, cost)
    
    sharpe = ret.mean()/ret.std()*np.sqrt(24*365)
    
    down = max_drawdown(np.cumprod(1+ret))
    
    return ret, sharpe, down, anticor.weight_lst



def batch_fixed (sharp_dat, dat, train_size, window_lst, holding_lst, init_weight, cost):
    '''
    Run a batch of window_lst and holding_lst result apply to the anticor_fixed
    AR algorithm
    
    Parameters:
    ----
    sharp_dat  :pd.DataFrame
        sharp ratio data
    dat        :pd.DataFrame
        price dat
    train_size :int
        size for training the filter
    window_lst :np.array
        window size for observing the anticorrelation
    holding_lst:np.array
        holding period for assets
    init_weight:np.array
        initial weight for the assets     
    cost       :float
        cost rate for each trade
    
    Return:
    ----
    pd.DataFrame:
        Sharpe ratio
    pd.DataFrame:
        Drawdown
    '''
    
    
    sharpe_lst = []
    down_lst = []
    for window in window_lst:
        temp1 = []
        temp2 = []
        for holding in holding_lst:
            _, sharpe, down, _ = anticor_algo_fixedAR(sharp_dat, dat, train_size, window, 
                                                   holding, init_weight, cost)
            print(sharpe)
            temp1.append(sharpe)
            temp2.append(down)
        sharpe_lst.append(temp1)
        down_lst.append(temp2)
        
    res1 = pd.DataFrame(sharpe_lst)
    res1.index = window_lst
    res1.columns = holding_lst
    
    res2 = pd.DataFrame(down_lst)
    res2.index = window_lst
    res2.columns = holding_lst
    
    return res1, res2

def batch_dynamic(sharp_dat, dat, train_size, train_window, predict_size, 
                                window_lst, holding_lst, init_weight,cost):
    
      
    '''
    Run a batch of window_lst and holding_lst result apply to the anticor_algo
    _dynamicAR algorithm
    
    
    Parameters:
    ----
    sharp_dat   :pd.DataFrame
        sharp ratio data
    dat         :pd.DataFrame
        price dat
    train_size  :int
        size for training the filter
    train_window:int
        size for training the AR coefficients
    predict_size:int
        size for using the AR coefficients
    window_lst  :np.array
        window size for observing the anticorrelation
    holding_lst :np.array
        holding period for assets
    init_weight :np.array
        initial weight for the assets     
    cost        :float
        cost rate for each trade
        
    Return:
    ----
    pd.DataFrame:
        Sharpe ratio
    pd.DataFrame:
        Drawdown
    '''
    
    
    sharpe_lst2 = []
    down_lst2 = []
    
    for window in window_lst:
        temp1 = []
        temp2 = []
        for holding in holding_lst:
            _, sharpe, down, _ = anticor_algo_dynamicAR(sharp_dat, dat, train_size, 
                train_window, predict_size, window, holding, init_weight, cost)
            print(sharpe)
            temp1.append(sharpe)
            temp2.append(down)
            
        sharpe_lst2.append(temp1)
        down_lst2.append(temp2)
    
    res1 = pd.DataFrame(sharpe_lst2)
    res1.index = window_lst
    res1.columns = holding_lst
    
    res2 = pd.DataFrame(down_lst2)
    res2.index = window_lst
    res2.columns = holding_lst    
    return res1, res2






