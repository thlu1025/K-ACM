# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:54:58 2019

@author: lth
"""
import numpy as np
import copy 
import pandas as pd

class Anticor:
    CAPR_df = None 
    price_df = None
    Claim = None
    wprev = None
    wnow = None
    window = None
    holding = None
    weight_lst = None
    dim = None
    
    def __init__(self, window, holding, CAPR_df, price_df, init_weight):
        '''
        
        Parameters:
        -----
        window     : int
            window size for measuring correlation
        holding    : int
            holing period for the strategy
        CAPR_df    : pd.DataFrame
            denoised price data frame time series
        price_df   : pd.DataFrame
            the original price dataframe
        init_weight: np.array
            initial weight for the strategy 
        '''
        self.CAPR_df = CAPR_df
        self.wprev = init_weight
        self.window = window
        self.holding = holding
        self.weight_lst = []
        self.dim = CAPR_df.shape[1]
        self.price_df = price_df

    def getClaim(self, Z1_df, Z2_df, n, strategy):
        '''      
        Pass in the CAPR price for certain window. Modify the claim matrix in 
        the class(not sure about the momentum part)
        
        Parameters:
        ----
        Z1_df, Z2_df: dataframe
            The dataframe for the CAPR price in certain window
        n           : int 
            Dimension of the asset
        strategy    : str 
            "mean_reversion", "momentum", "both"
        

        '''
    
        #Initialize the cross covariance matrix and the claim matrix
        Mcor = np.array([[None for i in range(n)] for j in range(n)])
        Claim = np.array([[0. for i in range(n)]for j in range(n)])
    
        #Mean and std of the latest window
        mu2 = np.mean(Z2_df, axis =0)
            
        #Calculate the cross correlation 
        for i in range(n):
            for j in range(n):
                Mcor[i,j] = np.corrcoef(Z1_df.iloc[:,i], Z2_df.iloc[:,j])[0][1]
    
        #Calculate the claim matrix
        for i in range(n):
            for j in range(n):
                if strategy == "mean_reversion":
                    if mu2[i]>= mu2[j] and Mcor[i,j]>0:
                        Claim[i,j] = Mcor[i,j] + max(-Mcor[i,i],0) 
                        + max (-Mcor[j,j] ,0)
                        
                elif strategy == "momentum":
                    if mu2[i] >= mu2[j] and Mcor[i,j]<=0:
                        Claim[i,j] = -Mcor[i,j] + max(Mcor[i,i],0) 
                        + max (Mcor[j,j] ,0)
                        
                elif strategy == "both":
                    if mu2[i]>= mu2[j] and Mcor[i,j]>0:
                        Claim[i,j] = Mcor[i,j] + max(-Mcor[i,i],0) 
                        + max (-Mcor[j,j] ,0)
                        
                    elif mu2[i] >= mu2[j] and Mcor[i,j]<=0:
                        Claim[j,i] = -Mcor[i,j] + max(Mcor[i,i],0) 
                        + max (Mcor[j,j] ,0)
                     
                    
        self.Claim = Claim
        
        
    def transweight(self, mkt_price):
        '''
        Transform the weight by using the claim matrix, normalizing the weight
        
        Parameters:
        ----
        mkt_price: pd.DataFrame
            The market price of the coins         
        
        '''
        
        transfer = copy.deepcopy(self.Claim)
        row, col = self.Claim.shape
        wnow = copy.deepcopy(self.wprev)
        for i in range(row):
            for j in range(col):
                if np.sum(self.Claim, axis = 1)[i] != 0:
                    transfer[i,j] = self.wprev[i]*self.Claim[i,j]/np.sum(self.Claim, axis =1)[i]
                
                else:
                    transfer[i,j] = 0
    
        for i in range(len(self.wprev)):
            wnow[i] = self.wprev[i] + np.sum(transfer, axis =0)[i] - np.sum(transfer, axis = 1)[i]
        
        #Normalize the weight
        wnow = (mkt_price.values * wnow)/np.dot(mkt_price.values, wnow)
        
        self.wnow = wnow
        
        
    def getweight(self, strategy):
        '''
        Balanced the weight by observing certain window, flip the weight if 
        both filtered mean price goes down, weight is balanced at the end 
        timestamp
        
        Parameters:
            ----
        strategy: str
            "momentum", "mean_reversion", "both"
        
        
        '''
        temp = 0
        for i in range(2*self.window):  
            self.weight_lst.append(self.wprev)
        

        
        for i in range(2*self.window, len(self.CAPR_df)):
            if i < temp + self.holding:
                self.weight_lst.append(self.wprev)
            else:
                #from timestamp 2w, esitmate the weight from timestamp 2*w+1 onwards
                Z1_df = self.CAPR_df.iloc[i-2*self.window: i-self.window,:]
                Z2_df = self.CAPR_df.iloc[i-self.window: i,:]
                
                row, col = Z1_df.shape
                
                self.getClaim(Z1_df, Z2_df, col, strategy)

                mu1 = np.mean(Z1_df, axis =0)
                mu2 = np.mean(Z2_df, axis =0)
                
    
                self.transweight(self.price_df.iloc[i,:])
                
                if all(x < 0 for x in mu2-mu1):
                    self.weight_lst.append(-self.wnow)
                    
                else:
                    self.weight_lst.append(self.wnow)
                    
                self.wprev = self.wnow
                temp = i
                
                
        self.weight_lst= pd.DataFrame(self.weight_lst)
        self.weight_lst.columns = self.CAPR_df.columns
        self.weight_lst.index = self.CAPR_df.index
        
    
    def calret(self, price_df, gamma):
        '''
        Calculate the return based on the weight_lst in the object and the 
        price of the assets. (still need to modify the transcation cost code)
        
        Parameters:
        ----
            
        price_df    :pd.DataFrame
            The price dataframe for ETH and BTC
        gamma       :float
            Cost rate for each trade
            
        Return:
        ----
        pd.DataFrame:
            the return of the portfolio since 2017-05-01
        
        '''
        
        all_df = pd.concat([price_df, self.weight_lst], axis = 1).dropna()

        coin = all_df.iloc[:,:self.dim]
        ret = coin.diff().dropna()/coin.shift(1).dropna()
        weight = all_df.iloc[:,self.dim:].shift(1).dropna()
        
        
        ret_strats = np.sum(ret*weight, axis =1)
        cost = np.sum(gamma/2*np.abs(weight.diff().fillna(0)), axis =1)
        ret_strats = ret_strats*(1-cost)
        
        return ret_strats['2017-05-01':]
    
#    def calprof(self, price_df, gamma):
#        '''
#        price_df    :The price dataframe for ETH and BTC
#        gamma       :Cost rate for each trade
#        
#        '''
#        
#        all_df = pd.concat([price_df, self.weight_lst], axis = 1).dropna()
#        all_df.columns = ['ETH','BTC','ETH_w', 'BTC_w']
#        coin = all_df[['ETH','BTC']]
#        weight = all_df[['ETH_w', 'BTC_w']].shift(1).dropna()
#        weight.columns = ['ETH','BTC']
#        prof = np.sum(coin.diff().dropna()*weight)
#        
#        ret_strats = np.sum(ret*weight, axis =1)
#        cost = np.sum(gamma/2*np.abs(weight.diff().fillna(0)), axis =1)
#        ret_strats = ret_strats*(1-cost)
#        
#        return ret_strats['2017-05-01':]
    
    
    
    
    
    