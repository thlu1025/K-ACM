# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:55:12 2019

@author: lth
"""

from utils.anticor_script import *
from Class.dataimporter import DataImporter
from utils.dir import*

#import dat
header = r'E:\coventure\corr_model\data'
dt_import = DataImporter(header)
dt_import.importDat(r'/BTCUSD_1m.csv')
data1 = dt_import.getprice(60)
dt_import.importDat(r'/ETHUSD_1m.csv')
data2 = dt_import.getprice(60)

dat = pd.concat([data2, data1],axis =1).dropna()
dat.columns = ['ETH','BTC']


train_size = 24
train_window = 96
predict_size = 1
#window_lst = list(range(13,21))
#holding_lst = list(range(1,24))
window_lst = list(range(13,21))
holding_lst = list(range(1,6))
init_weight = np.array([0.5,0.5])
cost = 0.01*0.5



parent_path = getParent()
    
res1 = batch_fixed (dat, train_size, window_lst, holding_lst, init_weight, cost)
s1, d1 = res1
createDir(r'\res\2price\fixed')
s1.to_csv(parent_path + r'\res\2price\fixed' + r'\sharpe.csv')
d1.to_csv(parent_path + r'\res\2price\fixed' + r'\down.csv')

res2 = batch_dynamic (dat, train_size, train_window, predict_size, 
                                window_lst, holding_lst, init_weight,cost)

s2, d2 = res2
createDir('\res\2price\dynamic')
s2.to_csv(parent_path + r'\res\2price\dynamic' + r'\sharpe.csv')
d2.to_csv(parent_path + r'\res\2price\dynamic' + r'\down.csv')

