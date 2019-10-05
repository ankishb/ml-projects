
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


########## plot



########### BASIC CELL RNN
index_in_epoch = 0
perm_array = np.arange(x_train.shape[0])
np.random.shuffle(perm_array)

# get next batch
def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if index_in_epoch > x_train.shape[0]:
        start = 0 #start next batch
        index_in_epoch = batch_size
        
    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]


# parameters
n_steps = seq_len - 1
n_inputs = 4
n_neurons = 200
n_outputs = 4
n_layers = 2
learning_rate = 0.001
batch_size = 50
n_epochs = 100
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]


tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

# use BASIC RNN CELL
layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation= tf.nn.relu) \
          for layer in range(n_layers)]

# use Basic LSTM Cell 
#layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.elu)
#          for layer in range(n_layers)]

# use LSTM Cell with peephole connections
#layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons, 
#                                  activation=tf.nn.leaky_relu, use_peepholes = True)
#          for layer in range(n_layers)]

# use GRU cell
#layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu)
#          for layer in range(n_layers)]


##############
# first we had created layers independently
# now we are connecting that layers
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, x, dtype=tf.float32)

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
# keep only last step
outputs = outputs[:, n_steps-1, :]


# loss
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)


# run graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for iteration in range(int(n_epochs*train_set_size/batch_size)):
        
        x_batch, y_batch = get_next_batch(batch_size)
        sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
        
        if iteration % int(5*train_set_size/batch_size) == 0:
            # mean square error
            mse_train = loss.eval(feed_dict = {x: x_train, y: y_train})
            mse_valid = loss.eval(feed_dict = {x: x_valid, y: y_valid})
            
            print('%.2f epochs: MSE train/valid = %.6f/%.6f'\
                  %(iteration*batch_size/train_set_size, mse_train, mse_valid))
            
    y_train_pred = sess.run(outputs, feed_dict={x: x_train})
    y_valid_pred = sess.run(outputs, feed_dict={x: x_valid})
    y_test_pred = sess.run(outputs, feed_dict={x: x_test})
    
    

########## plotting
    
ft = 0 # 0 = open, 1 = close, 2 = highest, 3 = lowest

## show predictions
plt.figure(figsize=(15, 5));


plt.plot(np.arange(y_train.shape[0]), y_train[:,ft], color='blue', label='train target')

plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_valid.shape[0]), y_valid[:,ft],
         color='gray', label='valid target')

plt.plot(np.arange(y_train.shape[0]+y_valid.shape[0],
                   y_train.shape[0]+y_test.shape[0]+y_test.shape[0]),
         y_test[:,ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0]),y_train_pred[:,ft], color='red',
         label='train prediction')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_valid_pred.shape[0]),
         y_valid_pred[:,ft], color='orange', label='valid prediction')

plt.plot(np.arange(y_train_pred.shape[0]+y_valid_pred.shape[0],
                   y_train_pred.shape[0]+y_valid_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:,ft], color='green', label='test prediction')

plt.title('past and future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best');
plt.show()

plt.figure(figsize=(15, 5));

plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test.shape[0]),
         y_test[:,ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:,ft], color='green', label='test prediction')

plt.title('future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best');
plt.show()

corr_price_development_train = np.sum(np.equal(np.sign(y_train[:,1]-y_train[:,0]),
            np.sign(y_train_pred[:,1]-y_train_pred[:,0])).astype(int)) / y_train.shape[0]
corr_price_development_valid = np.sum(np.equal(np.sign(y_valid[:,1]-y_valid[:,0]),
            np.sign(y_valid_pred[:,1]-y_valid_pred[:,0])).astype(int)) / y_valid.shape[0]
corr_price_development_test = np.sum(np.equal(np.sign(y_test[:,1]-y_test[:,0]),
            np.sign(y_test_pred[:,1]-y_test_pred[:,0])).astype(int)) / y_test.shape[0]

print('correct sign prediction for close - open price for train/valid/test: %.2f/%.2f/%.2f'%(
    corr_price_development_train, corr_price_development_valid, corr_price_development_test))


        












