# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:08:19 2019

@author: lth
"""
import pandas as pd 
import numpy as np
class DataImporter:
    header = None
    address = None
    data = None 
    def __init__(self, header):
        '''
        Import the data by given a folder path
        
        Parameters:
        ----
        header: str
            the folder path of the files
        '''
        self.header = header
    
    def importDat(self, name):
        '''
        Import the file data into the object
        
        Parameters:
        ----
        name: str
            the file name of the file
                
        '''
        self.address = self.header + name
        self.data = pd.read_csv(self.address)
    
        
    def getlogprice (self, minute):
        '''
        Get the minute log price
        
        Parameters:
        ----
        minute: int 
            the minute needed for the price
        
        Return:
        ----
        pd.DataFrame:
            The log price from 2017-05-01 for given minute
            
        '''
        tempdf = self.data.set_index('datetime')
        tempdat = np.log(tempdf['close'][::minute])
        
        return tempdat.dropna()['2017-05-01':]
    
    def getprice (self,minute):
        '''
        Get the minute price
        
        Parameters:
        ----
        minute: int 
            the minute needed for the price
        
        Return:
        ----
        pd.DataFrame:
            The price from 2017-05-01 for given minute
            
        '''
        
        
        tempdf = self.data.set_index('datetime')
        tempdat = tempdf['close'][::minute]
        
        return tempdat.dropna()['2017-05-01':]
    
    def getreturn(self, minute):
        
        '''
        Get the minute return
        
        Parameters:
        ----
        minute: int 
            the minute needed for the return
        
        Return:
        ----
        pd.DataFrame:
            The return from 2017-05-01 for given minute
            
        '''
        
        
        
        tempdf = self.data.set_index('datetime')
        pricedat = tempdf['close'][::minute]
        retdat = pricedat.diff()/pricedat.shift(1)
        return retdat.dropna()['2017-05-01':]
    
    def getsharp(self, minute, ret_minute, measure_hour):
        '''
        Get the sharp ratio based on given data
        
        Parameters:
        ----
        minute         :the frequency of the given sharp ratio 
        ret_minute     :the frequency measure of the return    
        measure_hour   :how long data for measuring the sharp
        
        Return:
        ----
        pd.DataFrame:
            The sharp_ratio from 2017-05-01 for given minute
            
        
        '''
        
        tempdf = self.data.set_index('datetime')
        pricedat = tempdf['close'][::ret_minute]
        retdat = pricedat.diff()/pricedat.shift(1)
        sharp_dat = retdat.rolling(measure_hour*int(minute/ret_minute)).apply(lambda x: x.mean()/x.std())
        return sharp_dat[::int(minute/ret_minute)].dropna()['2017-05-01':]
           
    def sepDat(sep,data):
        '''
        Seperate the data into certain length of chunks
        
        Parameters:
        ----
        sep :int
            the certain length want to segement
            
        Return:
        ----
        list:
            the list of dataframe contain the segemented data
    
        '''
        
        
        lst = []
        for i in range(len(data)):
            if i %sep ==0 and i+sep <= len(data):
                temp = data.iloc[i:(i+sep)]
                lst.append(temp)
        return lst

    def sepDat2(window, sep, data):
        '''
        Segment data by certain length of window and sep
        
        Parameters:
        ----
        window  :int
            the fixed length for each dataframe in the list
        sep     :int
            the fixed difference for each start index of the dataframe in the list
        data    :pd.DataFrame
            the dataframe needed to be seperated
            
        Return:
        ----
        list:
            List of pandas dataframe segmented
        '''
        
        
        lst = []
        for i in range(len(data)):
            if i %sep ==0 and i+window <= len(data):
                temp = data.iloc[i:(i+window)]
                lst.append(temp)
        return lst
    
        
