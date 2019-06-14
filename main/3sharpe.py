# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:41:30 2019

@author: lth
"""

from utils.anticor_sharpe_script import *
from Class.dataimporter import DataImporter
from utils.dir import *


minute = 60
ret_minute = 60
window = 24

train_size = 24
train_window = 96
predict_size = 1
window_lst = list(range(13,21))
holding_lst = list(range(1,6))
#window_lst = [13]
#holding_lst = [1]
init_weight = np.ones(3)/3
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

dt_import.importDat(r'/LTCUSD_1h.csv')
raw   = dt_import.data.set_index('datetime')['close']
ret3 = (raw.dropna().diff().dropna()/raw.shift(1).dropna())
sharp3 = ret3.rolling(24).apply(lambda x: x.mean()/x.std()).dropna()
sharp3 = sharp3['2017-05-01':]
dat3 = raw['2017-05-01':]


sharp_dat = pd.concat([sharp3, sharp2, sharp1],axis =1).dropna()
sharp_dat.columns = ['LTC','ETH','BTC']
dat = pd.concat([dat3, dat2, dat1], axis =1).dropna()
dat.columns = ['LTC','ETH','BTC']



res1 = batch_fixed (sharp_dat, dat, train_size, window_lst, holding_lst, init_weight, cost)
parent_path = getParent()

s1, d1 = res1
createDir(r'\res\3sharpe\fixed')
s1.to_csv(parent_path + r'\res\3sharpe\fixed' + r'\sharpe.csv')
d1.to_csv(parent_path + r'\res\3sharpe\fixed' + r'\down.csv')


res2 = batch_dynamic (sharp_dat, dat, train_size, train_window, predict_size, 
                                window_lst, holding_lst, init_weight,cost)


s2, d2 = res2
createDir(r'\res\3sharpe\dynamic')
s2.to_csv(parent_path + r'\res\3sharpe\dynamic' + r'\sharpe.csv')
d2.to_csv(parent_path + r'\res\3sharpe\dynamic' + r'\down.csv')

