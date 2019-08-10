
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# problems = pd.read_csv('problems_features.csv')
# tags = problems['tags'].unique()

# [one_hot(d, 50) for d in tags[1:]]

# max_length = 4
# padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# print(padded_docs)


# In[ ]:


# max_length = 4
# padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# print(padded_docs)


# In[ ]:


# model = Sequential()
# e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
# model.add(e)
# model.add(Flatten())


# In[2]:


results = pd.read_csv('data/train/train_submissions.csv')
test = pd.read_csv('data/test_submissions_NeDLEvX.csv')
users = pd.read_csv('cleaned_users.csv')
problems = pd.read_csv('problems_features.csv')
results.shape, test.shape, problems.shape, users.shape


# In[173]:


test = pd.read_csv('data/test_submissions_NeDLEvX.csv')


# In[4]:


users.head()


# In[5]:


problems.columns


# In[6]:


users.columns


# In[ ]:


temp = pd.concat([test, results], axis=0, ignore_index=True)
df.pivot_table('col3',index='col1',columns='col2')
np.argwhere(df.notnull().values)


# In[7]:


users.head()


# ### Encoding all user and prob ids

# In[175]:


from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

u_ids = list(users.user_id.values) + list(results.user_id.values) + list(test.user_id.values)
p_ids = list(problems.problem_id.values) + list(results.problem_id.values) + list(test.problem_id.values)

enc.fit(u_ids)
users['user_id_enc'] = enc.transform(users.user_id)
results['user_id_enc'] = enc.transform(results.user_id)
test['user_id_enc'] = enc.transform(test.user_id)

enc.fit(p_ids)
problems['problem_id_enc'] = enc.transform(problems.problem_id)
results['problem_id_enc'] = enc.transform(results.problem_id)
test['problem_id_enc'] = enc.transform(test.problem_id)


# In[14]:


cols = users.columns
cols


# In[87]:


# plt.plot(users[cols[idx]])
# plt.title(cols[idx])
# idx += 1


# ## Replacing the outliers with the median of that feature

# In[85]:


cols = ['submission_count', 'problem_solved',
       'contribution_abs', 'follower_count', 'max_rating', 'rating']
for idx in cols:
    mean, std = users[idx].mean(), users[idx].std()
    median = users[idx].loc[(users[idx] - mean).abs() < 3*std].median()
    users[idx +'new'] = np.where((users[idx] - mean).abs() > 3*std, median,users[idx])
    


# In[80]:





# In[92]:


problems.columns


# In[86]:


users.head()


# In[98]:


problems['level_type'] = problems['level_type'].fillna('O')


# In[99]:


enc.fit(problems['level_type'])
problems['level_type_enc'] = enc.transform(problems['level_type'])


# In[100]:


problems.head()


# In[105]:


problems['tags'].fillna('The')[:5]


# In[103]:


problems['tags_list'] = problems['tags'].fillna('The')


# ## convert tags into sequnce

# In[106]:


from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences


# In[114]:


tags = problems['tags_list']#.unique()

[one_hot(d, 50) for d in tags][:6]


# In[121]:


problems['tags_list_seq'] = problems['tags_list'].map(lambda x: one_hot(x,10))


# In[ ]:


pad_sequences([one_hot(d, 50) for d in tags], maxlen=5, padding='post')


# In[115]:


pads = pad_sequences([one_hot(d, 50) for d in tags], maxlen=5, padding='post')[:6]


# In[141]:


pads = pad_sequences([one_hot(d, 50) for d in tags], maxlen=5, padding='post')


# In[144]:


pads_list = []
for i in range(pads.shape[0]):
#     print(pads[i])
    pads_list.append(list(pads[i]))


# In[150]:


test = pd.Series(pads_list)
problems['tags_list_seq_pad'] = test.values


# In[154]:


problems.head()


# ### Drop unnecaasry columns

# In[185]:


import gc
del users,problems,test,results,testing
gc.collect()


# In[193]:


users = pd.read_csv( 'users_feature_encoded_before_drop.csv')
problems = pd.read_csv( 'problems_feature_encoded_before_drop.csv')
results = pd.read_csv( 'results_feature_encoded_before_drop.csv')
tests = pd.read_csv( 'test_feature_encoded_before_drop.csv')


# In[200]:


tests.columns


# In[202]:


u_ids = ['Unnamed: 0', 'Unnamed: 0.1', 'user_id', 'submission_count',
       'problem_solved', 'contribution', 'follower_count', 'max_rating',
       'rating', 'rank', 'contribution_abs','testing']
p_ids = ['Unnamed: 0', 'Unnamed: 0.1', 'problem_id', 'level_type', 'points',
         'tag_dict', 'tag_vect','tags_list', 'tags_list_seq']
r_ids = ['Unnamed: 0', 'user_id', 'problem_id']
t_ids = ['Unnamed: 0', 'ID', 'user_id', 'problem_id']


# In[203]:


users.drop(u_ids, axis=1, inplace=True)
problems.drop(p_ids, axis=1, inplace=True)
results.drop(r_ids, axis=1, inplace=True)
tests.drop(t_ids, axis=1, inplace=True)


# In[205]:


users.columns.shape, problems.columns.shape, results.columns.shape, tests.columns.shape


# In[206]:


users.head()


# In[207]:


results.head()


# In[209]:


rating_temp = pd.merge(results,users,on='user_id_enc',how='inner')


# In[211]:


rating_temp.shape


# In[212]:


rating_temp2 = pd.merge(rating_temp, problems,on='problem_id_enc',how='inner')


# In[213]:


rating_temp2.head()


# In[215]:


rating_temp2.to_csv('merged_all_feature_encoded_after_drop.csv',index=None)


# In[216]:


rating_temp2.isnull().sum()


# In[218]:


rating_temp2.user_id_enc.unique().shape, rating_temp2.problem_id_enc.unique().shape


# In[222]:


rating_temp3 = rating_temp2.dropna()


# In[221]:


rating_temp2.shape,rating_temp2.dropna().shape


# In[223]:


rating_temp3.to_csv('merged_all_feature_encoded_after_drop_non_null.csv',index=None)


# In[230]:


rating_temp3.values[:,:3]


# In[233]:


# del users, problems, results, tests, rating_temp, rating_temp2, rating_temp3
# gc.collect()


# In[231]:


data = rating_temp3


# In[249]:


data.columns


# ## dataset generate function

# In[293]:


def generate_dataset(dataset, batch_size):
    ''' return dataset for users, problems, users-info, 
        dataset has columns: ['attempts_range', 'user_id_enc', 'problem_id_enc', 'rank_enc',
                               'submission_countnew', 'problem_solvednew', 'contribution_absnew',
                               'follower_countnew', 'max_ratingnew', 'ratingnew', 'tags',
                               'level_type_enc', 'tags_list_seq_pad'], dtype='object')
    users    ==>> [batch,N]
    problems ==>> [batch,M]
    user-info==>> [batch,omega_u]
    prob-info==>> [batch,omega_v]
    label    ==>> [batch,label]
    '''
    N = dataset['user_id_enc'].unique().shape[0]
    M = 5776#dataset['problem_id_enc'].unique().shape[0]
    label_shape = dataset['attempts_range'].unique().shape[0]
    
    idx = np.random.randint(dataset.shape[0],size=(batch_size))
    temp = dataset.values[idx]
    users = temp[:,:3]# first three columns
    users_info = temp[:,3:10]
    problems_info = temp[:,11:]
    problems_info[:,1] = problems_info[:,1].astype(int)
    
    # user one-hot matrix
    users = np.zeros((batch_size,N))
    users[np.arange(batch_size), temp[:,1].astype(int)[0]] = 1  #one-hot vectors
    
    # problems one-hot matrix
    problems = np.zeros((batch_size,M))
    problems[np.arange(batch_size), temp[:,2].astype(int)[0]] = 1
    
    # label one-hot matrix
    labels = np.zeros((batch_size,label_shape))
    labels[np.arange(batch_size),temp[:,0].astype(int)[0]] = 1
                          
    return users, problems, labels, users_info, problems_info


# In[294]:


users, problems, labels, users_info, problems_info = generate_dataset(dataset=data, batch_size=2)
users.shape, problems.shape, labels.shape, users_info.shape, problems_info.shape


# In[310]:


# Convert a sequence of object data-type into array  of int data-type

# I have array of this following shape and dtype:

#     array([[1, '[13, 39, 0, 0, 0]'],
#            [2, '[9, 31, 5, 0, 0]']], dtype=object)

# I want it to merge both into one simple array with float dtype 

#     array([[1, 13, 39, 0, 0, 0],
#            [2, 9, 31, 5, 0, 0]], dtype=float)

# Please, if someone can help me?


# In[312]:


get_ipython().run_cell_magic('file', 'final_pipeline.py ', '')


# In[ ]:


from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Activation, Dense, Dropout
from keras.layers import concatenate

import numpy as np
import tensorflow as tf


# ## define parameters

# In[2]:


N = 5
M = 3
K = 2
N_u = 5
M_v = 6


# ## define a_u and a_v as the tensor, because we are not sampling anything

# In[3]:


X = np.random.randn(N,M)
a_u = np.random.randn(N,N_u)
a_v = np.random.randn(M,M_v)

# X = tf.convert_to_tensor(X, dtype=tf.float32)
a_u = tf.convert_to_tensor(a_u, dtype=tf.float32)
a_v = tf.convert_to_tensor(a_v, dtype=tf.float32)


def next_batch(batch_size):
    '''
    returnn data of shape [ [batch,N], [batch,M], [batch,1] ]
    fetch index from the matrix of shape [batch,batch]
    '''
    x_idx = np.random.randint(N,size=(batch_size))
    y_idx = np.random.randint(M,size=(batch_size))
    
    rating_label = []
    for i,j in zip(x_idx,y_idx):
        rating_label.append(X[i][j])
#     rating_label = tf.convert_to_tensor(rating_label,dtype=tf.float32)
#     rating_label = tf.reshape(rating_label,shape=[-1,1])
    
#     return Xn, Xm, rating_label
    return x_idx, y_idx, rating_label


# ## build model

# In[4]:


input_user = tf.placeholder(tf.int32,shape=(None,),name='batch-user')
input_user_hot = tf.one_hot(input_user,N, name='one-hot-user')
user_latent = Dense(K,activation='linear', name='user-latent')(input_user_hot)

input_item = tf.placeholder(tf.int32,shape=(None,),name='batch-item')
input_item_hot = tf.one_hot(input_item,M, name='one-hot-item')
item_latent = Dense(K,activation='linear', name='item-latent')(input_item_hot)

info_user = tf.matmul(input_user_hot, a_u)
info_user_latent = Dense(K,activation='linear',name='info-user-latent')(info_user)

info_item = tf.matmul(input_item_hot, a_v)
info_item_latent = Dense(K,activation='linear', name='item-user-latent')(info_item)


# In[5]:


input_user, user_latent, input_item, item_latent, info_user_latent, info_item_latent


# In[6]:


pred_rating = tf.matmul(user_latent,tf.transpose(item_latent))#,transpose_item_latent)
pred_rating = tf.diag_part(pred_rating)
pred_rating = tf.reshape(pred_rating,shape=[-1,1])

pred_rating_content = tf.matmul(info_user_latent,tf.transpose(info_item_latent))#,transpose_info_item_latent)
pred_rating_content = tf.diag_part(pred_rating_content)
pred_rating_content = tf.reshape(pred_rating_content, shape=[-1,1])

final_rating = pred_rating + pred_rating_content


# In[7]:


final_rating


# ## loss function (mse. no classification)

# In[8]:


actual = tf.placeholder(tf.float32, shape=(None,),name='rating')

loss = tf.reduce_mean(tf.square(pred_rating - actual))
loss_user = tf.reduce_mean(tf.square(user_latent - info_user_latent))
loss_item = tf.reduce_mean(tf.square(item_latent - info_item_latent))
final_loss = loss + loss_user + loss_item


# In[9]:


final_loss


# In[10]:


optimizer = tf.train.AdamOptimizer(0.001).minimize(final_loss)

correct_pred = tf.equal(final_rating, actual)
correct_pred = tf.cast(correct_pred, tf.float32)
accuracy = tf.reduce_mean(correct_pred)

accuracy


# In[11]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 2
for i in range(10):
    batch = next_batch(batch_size)
    feed_data = {input_user : batch[0], input_item : batch[1], actual : batch[2]}
#     if i % 2 == 0:
#         train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
#         print('step %d, training accuracy %g' % (i, train_accuracy))
        
    loss_iter = sess.run(final_loss,feed_dict=feed_data)
    print('step %d, training accuracy %g' % (i, loss_iter))

    sess.run(optimizer, feed_dict=feed_data)

