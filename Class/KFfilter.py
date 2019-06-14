# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:25:48 2019

@author: lth
"""
import numpy as np
from statsmodels.tsa.api import VAR
import pandas as pd
from Class.dataimporter import DataImporter

class KFfilter:
    
    data = None
    train_size = None
    train_window = None
    predict_size = None
    init_val = None
    init_prob = None
    R = None
    Q = None
    dim = None 
          
    def __init__(self, data, train_size, train_window, predict_size):
        '''
        Kalman filter implmentation for fixed coef and dynamic coef
        
        Parameters:
        ----
        data        : pd.DataFrame
            Input data for filtering
        train_size  : int
            Training size for getting filtering parameters
        train_window: int
            The window size for training the VAR model
        predict_size: int
            The size of the prediction for using the previous VAR model
        
        '''
        self.data = data
        self.train_size = train_size
        self.train_window = train_window
        self.predict_size = predict_size
        self.dim = data.shape[1]    
        
    def trainParam (self):
        '''
        From 2017-05-01 
        Using fixed trans_mat and obs_mat to train to get the two noise covariance
        matrix and the initial covariance matrix
        
        '''
        train_dat = self.data['2017-05-01 01:00:00':].iloc[:self.train_size,:]
        trans_mat = np.eye(self.dim)
        obs_mat = np.eye(self.dim)
        init = train_dat.iloc[-1,:]
        
        from pykalman import KalmanFilter
        kf = KalmanFilter(
              transition_matrices = trans_mat,
              observation_matrices = obs_mat,
              initial_state_mean = init,
              em_vars = [ 'observation_covariance',
                         'transition_covariance','initial_state_covariance']
            )
        init_filter = kf.em(train_dat, n_iter =3)
        x, p = init_filter.filter(train_dat)
        
        self.init_val = x[-1]
        self.init_prob = p[-1]
        self.R = init_filter.observation_covariance
        self.Q = init_filter.transition_covariance
    



    def fixed_filter(dat, train_size, trans_mat, obs_mat):
        '''
        External function for fixed transition matrix filter usage
        Fixed transition matrix and observation matrix 
        
        Parameters:
        ----
        dat       : pd.DataFrame
            the input data for filtering
        train_size: int
            the training size to get the initial parameters
        trans_mat : np.array
            the transition matrix for fixed implmentation of kalman filter
        obs_mat   : np.array
            the observation matrix 
        
        Return:
        ----
        pd.DataFrame
            The filtered result in pandas dataframe, [:dim] is filtered res
            [dim:] is real price
        '''
        from pykalman import KalmanFilter
       
        train_dat = dat.iloc[:train_size,:]
        measure_dat = dat.iloc[train_size:,:]
        init = dat.iloc[train_size-1,:]        
        kf =  KalmanFilter(
                      transition_matrices = trans_mat,
                      observation_matrices = obs_mat,
                      initial_state_mean = init,
                      em_vars = [ 'observation_covariance',
                                 'transition_covariance',
                                 'initial_state_covariance']
                          ) 
        trained_filter = kf.em(train_dat, n_iter =3)
        x,p = trained_filter.filter(measure_dat)
        
        temp = pd.DataFrame(x)
        temp.columns = dat.columns
        temp.index = measure_dat.index
        
        res = KFfilter.merge(temp, measure_dat)
        return res
        
    def merge(temp1, temp2):
        '''
        Merge the filtered result and measurement result 
        
        Parameters:
        ----
        temp1: pd.DataFrame 
            filtered 
        temp2: pd.DataFrame
            measurement
            
        Return:
        ----
        pd.DataFrame
            the merged dataframe
        '''
        temp = pd.concat([temp1, temp2], axis =1 ).dropna()
        return temp
        
    def adapt_filter(self, obs_mat):
        '''
        Adaptive filter by using a fixed observation matrix and dynamic transition
        matrix
        
        Parameters:
        ----
        obs_mat         :np.array() 
            the fixed observation matrix
        Return:
        ----
            the filtered result in dataframe
        '''
        from filterpy.kalman import KalmanFilter
        measure_dat = self.data['2017-05-01 01:00:00':].iloc[self.train_size:,:]
        #seperating lists
        train_lst = DataImporter.sepDat2(self.train_window, self.predict_size, measure_dat)
        measure_lst = DataImporter.sepDat(self.predict_size, measure_dat.iloc[self.train_window:,:])
        #estimating coefficients
        import warnings
        warnings.filterwarnings("ignore")
        param_lst =  [VAR(i).fit(1, trend = 'nc').params.values for i in train_lst]
        
        #calculating the kf results 
        res_lst = []
        #initialize filter params
        kf = KalmanFilter(dim_x = self.dim, dim_z = self.dim)
        kf.x = self.init_val
        kf.P = self.init_prob
        kf.R = self.R
        kf.Q = self.Q
        kf.H = obs_mat
        #filtering 
        for i in range(len(measure_lst)):
            #changing params of transition mat
            kf.F = param_lst[i].T
            #doing one batch 
            x, p, _, _ = kf.batch_filter(measure_lst[i].values)
            res = pd.DataFrame(x)
            res.columns = self.data.columns
            res.index = measure_lst[i].index
            res_lst.append(res)
            
        return KFfilter.merge(pd.concat(res_lst), measure_dat)
        
        

        
        
        
        
        
        