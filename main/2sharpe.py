# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:57:36 2019

@author: lth
"""

from utils.anticor_sharpe_script import *
from Class.dataimporter import DataImporter
from utils.dir import *
#parameters
#sharp ratio measuring parameters
minute = 60
ret_minute = 10
window = 6

train_size = 24
train_window = 96
predict_size = 1
window_lst = list(range(13,21))
holding_lst = list(range(1,6))
#window_lst = [16]
#holding_lst = [1]
init_weight = np.array([0.5,0.5])
cost = 0.01*0.5
    




#import dat
header = r'E:\coventure\corr_model\data'
dt_import = DataImporter(header)
dt_import.importDat(r'/BTCUSD_1m.csv')

sharp1 = dt_import.getsharp(minute, ret_minute, window)
dat1 = dt_import.getprice(minute)

dt_import.importDat(r'/ETHUSD_1m.csv')
sharp2 = dt_import.getsharp(minute, ret_minute, window)
dat2 = dt_import.getprice(minute)

sharp_dat = pd.concat([sharp2, sharp1],axis =1).dropna()
sharp_dat.columns = ['ETH','BTC']
dat = pd.concat([dat2, dat1], axis =1).dropna()
dat.columns = ['ETH','BTC']



res1 = batch_fixed (sharp_dat, dat, train_size, window_lst, holding_lst, init_weight, cost)
parent_path = getParent()

s1, d1 = res1
createDir(r'\res\2sharpe\fixed')
s1.to_csv(parent_path + r'\res\2sharpe\fixed' + r'\sharpe.csv')
d1.to_csv(parent_path + r'\res\2sharpe\fixed' + r'\down.csv')



res2 = batch_dynamic (sharp_dat, dat, train_size, train_window, predict_size, 
                                window_lst, holding_lst, init_weight,cost)


s2, d2 = res2
createDir(r'\res\2sharpe\dynamic')
s2.to_csv(parent_path + r'\res\2sharpe\dynamic' + r'\sharpe.csv')
d2.to_csv(parent_path + r'\res\2sharpe\dynamic' + r'\down.csv')


