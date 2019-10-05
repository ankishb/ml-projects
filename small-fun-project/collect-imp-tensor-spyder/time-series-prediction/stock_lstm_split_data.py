
# In[]: import libraries

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
print("hello")

# In[]: load dataset

file_path = 'D:\Tensorflowspyder\Time_series_predition\dataset.csv'
data = pd.read_csv(file_path, index_col=0)


# In[]: data normalization

def normalization(data):
    norm = MinMaxScaler()
    data['open'] = norm.fit_transform(data.open.values.reshape(-1,1))
    data['low'] = norm.fit_transform(data.low.values.reshape(-1,1))
    data['high'] = norm.fit_transform(data.high.values.reshape(-1,1))
    data['close'] = norm.fit_transform(data.close.values.reshape(-1,1))
    return data

data_columns = data.columns.values
print(data_columns)

# In[]: make array of 4 columns for 'YHOO'

data_stock = data[data.symbol == 'YHOO']
data_stock.drop(['symbol'], 1, inplace=True)
data_stock.drop(['volume'], 1, inplace=True)

data_stock_norm = normalization(data_stock)

# In[]: prepare train and test dataset

def prepare_data(data, seq_length):
    seq_list = []
    for i in range(len(data) - seq_length):
        temp = data[i:i+seq_length]
        seq_list.append(temp)
    return seq_list

seq = prepare_data(data_stock_norm, 10)

# In[]:

# stock kaggle.py

import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf
print('hello')
# split data in 80%/10%/10% train/validation/test sets
valid_set_size_percentage = 10 
test_set_size_percentage = 10 

##display parent directory and working directory
#print(os.path.dirname(os.getcwd())+':', os.listdir(os.path.dirname(os.getcwd())))
#print(os.getcwd()+':', os.listdir(os.getcwd()))

file_path = 'tensorflow_data/prices-split-adjusted.csv'
df = pd.read_csv(file_path, index_col = 0)
df.info()
#df.head()

# In[]
######## Manipulated data

def normalize_data(df):
    min_max_scalar = sklearn.preprocessing.MinMaxScaler()
    df['open'] = min_max_scalar.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scalar.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scalar.fit_transform(df.low.values.reshape(-1,1))
    df['close'] = min_max_scalar.fit_transform(df.close.values.reshape(-1,1))
    
    return df

######### prepare train test validational dataset
    
def load_data(stock, seq_len):
    data_raw = stock.as_matrix()
    data = []
    
    for index in range(len(data_raw) - seq_len):
        data.append(data_raw[index:index+seq_len])
        
    data = np.array(data)
    print(data.shape)
    valid_set_size = int(np.round(valid_set_size_percentage/100 * data.shape[0]))
    test_set_size = int(np.round(test_set_size_percentage/100 * data.shape[0]))
    train_set_size = data.shape[0] - valid_set_size - test_set_size
    print(data[:1])
    x_train = data[:train_set_size, :-1, :]############
    print(x_train[:1])
    y_train = data[:train_set_size, -1, :]#########check
    
    x_valid = data[train_set_size:train_set_size+valid_set_size, :-1, :]
    y_valid = data[train_set_size:train_set_size+valid_set_size, -1, :]
    
    x_test = data[train_set_size+valid_set_size:, :-1, :]
    y_test = data[train_set_size+valid_set_size, -1, :]
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test

# choose one stock
df = data  
df_stock = df[df.symbol == 'EQIX'].copy()
df_stock.drop(['symbol'], 1, inplace=True)
df_stock.drop(['volume'], 1, inplace=True)


cols = list(df_stock.columns.values)
print('columns name :', cols)


# normalize stock
df_stock_norm = df_stock.copy()
df_stock_norm = normalize_data(df_stock_norm)


# create train test valid dataset
seq_len = 20
x_train, y_train, x_valid, y_valid, x_test, y_test = \
load_data(df_stock_norm, seq_len)

print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid.shape = ',x_valid.shape)
print('y_valid.shape = ', y_valid.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ',y_test.shape)

