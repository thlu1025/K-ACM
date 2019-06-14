# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 23:49:25 2019

@author: lth
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:58:51 2019

@author: lth
"""

from utils.anticor_script import *
from Class.dataimporter import DataImporter
from utils.dir import *

#import dat
header = r'E:\coventure\corr_model\data'
dt_import = DataImporter(header)
dt_import.importDat(r'/BTCUSD_1m.csv')
data1 = dt_import.getprice(60)
dt_import.importDat(r'/ETHUSD_1m.csv')
data2 = dt_import.getprice(60)
dt_import.importDat(r'/LTCUSD_1h.csv')
data3 = dt_import.data.set_index('datetime')['close']['2017-05-01':]



dat = pd.concat([data3, data2],axis =1).dropna()
dat.columns = ['LTC','ETH']


train_size = 24
train_window = 96
predict_size = 1
#window_lst = list(range(13,21))
#holding_lst = list(range(1,24))
window_lst = list(range(13,21))
holding_lst = list(range(1,6))
init_weight = np.array([0.5,0.5])
cost = 0.01*0.5