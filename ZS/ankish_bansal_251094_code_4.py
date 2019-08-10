
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import os, gc
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 200)
pd.set_option('display.max.columns', 200)


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

sns.set(context='notebook', style='whitegrid', palette='deep', font='sans-serif', 
        font_scale=2, color_codes=True, rc=None)


# In[2]:


df = pd.read_csv('data/data.csv')
sub = pd.read_csv('data/sample_submission.csv')
df.head()


# In[5]:


# df[df.is_goal.isnull()].shot_id_number.isnull().sum()


# In[6]:


df = df[~df.shot_id_number.isnull()]


# In[7]:


df.info()


# In[8]:


df.drop(['Unnamed: 0', 'team_name','team_id'], axis=1, inplace=True)
df.shape


# In[9]:


for col in df.columns:
    print("{0:<25} {1:<10} {2}".format(col, df[col].unique().shape[0], df[col].dtype))


# In[11]:


df['date_of_game'] = df['date_of_game'].astype('str')
df['date_of_game'] = df['date_of_game'].apply(lambda x: x.replace('nan', '2016-05-01'))

print(df.shape, "==>", end=" ")
df['year']  = df['date_of_game'].apply(lambda x: x.split('-')[0])
df['month'] = df['date_of_game'].apply(lambda x: x.split('-')[1])
df.drop(['game_season', 'date_of_game'], axis=1, inplace=True)
print(df.shape)


# In[12]:


df['home/away'] = df['home/away'].astype('str')
df['home/away'] = df['home/away'].apply(lambda x: x.replace('nan', 'NEW'))

print(df.shape, "==>", end=" ")
df['home/away'] = df['home/away'].apply(lambda x: x[-3:])
print(df.shape)


# In[22]:


float_col = df.columns[df.dtypes == np.float64]
object_col = df.columns[df.dtypes == 'object']

float_col = list(float_col)
object_col = list(object_col)
float_col.remove('is_goal')

len(float_col), len(object_col)


# In[13]:


df[float_col].isnull().sum()


# In[15]:


# fig, ax = plt.subplots(figsize=(24, 4))

# g = sns.countplot(x='game_season', hue='is_goal', data=df, ax=ax)
# for item in g.get_xticklabels():
#     item.set_rotation(45)


# In[16]:


sns.scatterplot(df[df.is_goal == 0].location_y, df[df.is_goal == 0].location_x, alpha=0.8, color='c')
sns.scatterplot(df[df.is_goal == 1].location_y, df[df.is_goal == 1].location_x, alpha=0.3, color='k')


# In[17]:


sns.kdeplot(df[df.is_goal == 1].location_y)
sns.kdeplot(df[df.is_goal == 0].location_y)


# In[97]:


# from sklearn.cluster import KMeans

# kmeans = KMeans(n_clusters=10, init='k-means++', n_init=10, verbose=1, random_state=1337, n_jobs=4)
# kmeans.fit(df[['location_x','location_y']].dropna())


# In[98]:


# kmeans.cluster_centers_


# In[102]:


# fig, ax = plt.subplots(figsize=(22, 5))
# sns.scatterplot(x='location_x', y='location_y', data=df[['location_x','location_y']].dropna(), hue=kmeans.labels_, ax=ax)


# In[19]:


def get_quantile(df, col, q1, q2):
    """compute quantile range
    args:
        col: col name
        q1: lower quantile percentile
        q2: upper quantile percentile
    """
    df1 = df[[col]].dropna()
    lower_bound = np.percentile(df1, q=q1)
    upper_bound = np.percentile(df1, q=q2)
    lower_bound = np.round(lower_bound,3)
    upper_bound = np.round(upper_bound, 3)
    min_ = np.round(np.min(df1[col]), 3)
    max_ = np.round(np.max(df1[col]), 3)
    print("{4:<25} min: {0:<10} max: {1:<10} low: {2:<10} high: {3:<10}".format(min_, max_, lower_bound, upper_bound, col))

float_col.remove('match_event_id')
for col in float_col:
    try:
        get_quantile(df[df.is_goal == 0], col, 1, 99)
    except:
        print("couldn't do it")
        
print("=="*45)        

for col in float_col:
    try:
        get_quantile(df[df.is_goal == 1], col, 1, 99)
    except:
        print("couldn't do it")


# # SCORE: 1 / (1 + MAE)     .......  ##MAE: mean absolute error

# ## There seem to be duplicate columns, but turns out that these are uncorrelated component, which i guess, represent the first and second lap data, which make sense.

# In[20]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(18,15))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, ax=ax)


# In[23]:


fig, ax = plt.subplots(9,1, figsize=(25, 50))
axes = ax.flatten()

for i, col in enumerate(object_col):
    sns.countplot(x=col, hue='is_goal', data=df, ax=axes[i])
    print(i, end=" ")
    
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(30)


# In[24]:


df['lat/lng'] = df['lat/lng'].astype('str')
df['lat/lng'] = df['lat/lng'].apply(lambda x: x.replace('nan', '42.982923, -71.446094'))

print(df.shape, "==>", end=" ")
df['lat'] = df['lat/lng'].apply(lambda x: x.split(',')[0])
df['long'] = df['lat/lng'].apply(lambda x: x.split(',')[1])
df['lat'] = df['lat'].astype('float')
df['long'] = df['long'].astype('float')
print(df.shape)


# In[26]:


# Hierarchical clustering, PAM, CLARA, and DBSCAN are popular examples of this.
df['loc_x'] = np.cos(df['lat']) * np.cos(df['long'])
df['loc_y'] = np.cos(df['lat']) * np.sin(df['long'])
df['loc_z'] = np.sin(df['lat'])
df.shape


# In[27]:


df.isnull().sum().sort_values()


# In[28]:


# df['area_of_shot'] = df['area_of_shot'].apply(lambda x: x.replace('nan', 'Center(C)'))
df['area_of_shot'].fillna('Center(C)', inplace=True)
df['range_of_shot'].fillna('Unknown', inplace=True)
df['shot_basics'].fillna('Unknown', inplace=True)


# In[29]:


null_cols = ['location_x','power_of_shot','knockout_match.1','knockout_match','remaining_min.1','remaining_sec.1', 'distance_of_shot.1',
             'remaining_sec','power_of_shot.1','location_y','remaining_min','match_event_id','distance_of_shot']
df[null_cols].dtypes


# In[30]:


for col in null_cols:
    df[col] = df[col].fillna(df[col].median())


# In[31]:


df.isnull().sum()


# In[32]:


df['type_of_combined_shot'].fillna('shot - 6', inplace=True)
df['shot_null'] = df['type_of_shot'].isnull()
df['type_of_shot'].fillna('shot - 60', inplace=True)


# In[33]:


df.isnull().sum()


# In[34]:


df['loc_x'] = df['loc_x']*6371
df['loc_y'] = df['loc_y']*6371
df['loc_z'] = df['loc_z']*6371


# In[36]:


object_col = object_col + ['year','month']
for col in object_col:
    df[col] = df[col].astype('category').cat.codes


# In[37]:


df['shot_null'] = df['shot_null'].astype('int')
df.dtypes


# In[38]:


# sub1 = sub.copy()


# In[39]:


df['shot_id_number'] = df['shot_id_number'].astype(int)


# In[40]:


try:
    del df1
    gc.collect()
except:
    print("df not exist")
    
df1 = df.copy()
df1.isnull().sum()

train_df = df[~df.is_goal.isnull()]
test_df  = df[df.is_goal.isnull()]

target = train_df['is_goal']
train_df.drop(['is_goal'], axis=1, inplace=True)
test_df.drop(['is_goal'], axis=1, inplace=True)

train_df.shape, test_df.shape


# In[41]:


test_df_id = test_df.shot_id_number
len(test_df.shot_id_number)


# In[42]:


train_df.drop(['shot_id_number'], axis=1, inplace=True)
test_df.drop(['shot_id_number'], axis=1, inplace=True)


# In[43]:



import pandas as pd
import numpy as np
import os, gc
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from catboost import Pool, CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb


def train_lgb_model(X_train, y_train, X_valid, y_valid, features, param, X_test, num_round):
    """
    Args:
        X_train, X_valid: training and valid data
        y_train, y_valid: training and valid target
        X_test: test-data
        features: training features
    Return:
        oof-pred, test_preds model, model_imp
    """
    _train = lgb.Dataset(X_train[features], label=y_train, feature_name=list(features))
    _valid = lgb.Dataset(X_valid[features], label=y_valid,feature_name=list(features))
    
    clf = lgb.train(param, _train, num_round, 
                    valid_sets = [_train, _valid], 
                    verbose_eval=200, 
                    early_stopping_rounds = 25)                  
    
    oof = clf.predict(X_valid[features], num_iteration=clf.best_iteration)
    test_pred = clf.predict(X_test[features], num_iteration=clf.best_iteration)
    
    lgb_imp = pd.DataFrame(data=[clf.feature_name(), list(clf.feature_importance())]).T
    lgb_imp.columns = ['feature','imp']
    
    return oof, test_pred, clf, lgb_imp
    



def run_cv_lgb(train_df, target, test_df, leaves=None):

    param = {
        'bagging_freq'           : 5,
        'bagging_fraction'       : 0.33,
        'boost_from_average'     : 'false',
        'boost'                  : 'gbdt',
        'feature_fraction'       : 0.3,
        'learning_rate'          : 0.01,
        'max_depth'              : -1,
        'metric'                 : 'auc',
        'min_data_in_leaf'       : 100,
#         'min_sum_hessian_in_leaf': 10.0,
        'num_leaves'             : 30,
        'num_threads'            : 4,
        'tree_learner'           : 'serial',
        'objective'              : 'binary',
        'verbosity'              : 1,
    #     'lambda_l1'              : 0.001,
        'lambda_l2'              : 0.1
    }   
    if leaves is not None:
        param['num_leaves'] = leaves
        print("using leaves: ", param['num_leaves'])

    random_seed = 1234
    n_splits = 3
    num_round = 10000
    feature_imp = pd.DataFrame()
    
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof_lgb = np.zeros(len(train_df))
    predictions = np.zeros((len(test_df),n_splits))

    clfs = []
    
    for fold_, (train_index, valid_index) in enumerate(folds.split(train_df, target)):
        print(train_index.shape, valid_index.shape)
        print("Fold {}".format(fold_))
    
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        X_train, X_valid = train_df.iloc[train_index,:], train_df.iloc[valid_index,:]
        features = X_train.columns


        num_round = 10000
        oof, test_pred, clf, lgb_imp = train_lgb_model(X_train, y_train, 
                                                       X_valid, y_valid, 
                                                       features, param, 
                                                       test_df, num_round)
        lgb_imp['fold'] = fold_
        feature_imp = pd.concat([feature_imp, lgb_imp], axis=0)
    
        oof_lgb[valid_index] = oof
        predictions[:,fold_] = test_pred
        clfs.append(clf)
        
        score = mean_absolute_error(y_valid, oof)
        print( "  score = ", 1/(1 + score) )
        print("="*60)
    
    feature_imp.imp = feature_imp.imp.astype('float')
    feature_imp = feature_imp.groupby(['feature'])['imp'].mean()
    feature_imp = pd.DataFrame(data=[feature_imp.index, feature_imp.values]).T
    feature_imp.columns=['feature','imp']
    feature_imp = feature_imp.sort_values(by='imp')

    return clfs, feature_imp, oof_lgb, predictions



# In[44]:


clfs_lgb, imp_lgb, oof_lgb, pred_lgb = run_cv_lgb(train_df, target, test_df, leaves=50)


# In[45]:



fig, ax = plt.subplots(1,1,figsize=(18, 25))
sns.barplot(x='imp',y='feature',data=imp_lgb, ax=ax)


# In[46]:


len(list(set(train_df['match_id']).intersection(test_df['match_id']))), len(list(set(train_df['match_id']))), len(list(set(test_df['match_id'])))


# In[47]:


from catboost import Pool, CatBoostClassifier, CatBoostRegressor

def train_cat_model(X_train, y_train, X_valid, y_valid, features, param, X_test, 
                    num_round):
    """
    Args:
        X_train, X_valid: training and valid data
        y_train, y_valid: training and valid target
        X_test: test-data
        features: training features
    Return:
        oof-pred, test_preds, model, model_imp
    """
    param['iterations'] = num_round
    
    _train = Pool(X_train[features], label=y_train)#, cat_features=cate_features_index)
    _valid = Pool(X_valid[features], label=y_valid)#, cat_features=cate_features_index)

    watchlist = [_train, _valid]
    clf = CatBoostClassifier(**param)
    clf.fit(_train, 
            eval_set=watchlist, 
            verbose=200,
            use_best_model=True)
        
    oof  = clf.predict_proba(X_valid[features])[:,1]
    test_pred  = clf.predict_proba(X_test[features])[:,1]

    cat_imp = pd.DataFrame(data=[clf.feature_names_, 
                                 list(clf.feature_importances_)]).T
    cat_imp.columns = ['feature','imp']
    
    return oof, test_pred, clf, cat_imp


def run_cv_cat(train_df, target, test_df, depth):

    params = {
        'loss_function'         : "Logloss", 
        'eval_metric'           : "AUC",
        'random_strength'       : 1.5,
        'border_count'          : 128,
#         'scale_pos_weight'      : 3.507,
        'depth'                 : depth, 
        'early_stopping_rounds' : 50,
        'random_seed'           : 1337,
        'task_type'             : 'CPU', 
#         'subsample'             = 0.7, 
        'iterations'            : 10000, 
        'learning_rate'         : 0.09,
        'thread_count'          : 4
    }


    ##########################
    n_splits = 3
    random_seed = 1234
    feature_imp = pd.DataFrame()
    
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof_cat = np.zeros(len(train_df))
    predictions = np.zeros((len(test_df),n_splits))
    clfs = []
##########################
    for fold_, (train_index, valid_index) in enumerate(folds.split(train_df, target)):
        print(train_index.shape, valid_index.shape)
        print("Fold {}".format(fold_))
    
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        X_train, X_valid = train_df.iloc[train_index,:], train_df.iloc[valid_index,:]
        features = X_train.columns
        
        num_rounds = 10000
        oof, test_pred, clf, cat_imp = train_cat_model(X_train, y_train, 
                                                       X_valid, y_valid, 
                                                       features, params, 
                                                       test_df, num_rounds)
    
        oof_cat[valid_index] = oof
        predictions[:,fold_] = test_pred
        
        cat_imp['fold'] = fold_
        feature_imp = pd.concat([feature_imp, cat_imp], axis=0)
        clfs.append(clf)
        
        score = mean_absolute_error(y_valid, oof)
        print( "  score = ", 1/(1 + score) )
        print("="*60)
    
    feature_imp.imp = feature_imp.imp.astype('float')
    feature_imp = feature_imp.groupby(['feature'])['imp'].mean()
    feature_imp = pd.DataFrame(data=[feature_imp.index, feature_imp.values]).T
    feature_imp.columns=['feature','imp']
    feature_imp = feature_imp.sort_values(by='imp')

    return clfs, feature_imp, oof_cat, predictions


# In[48]:


clfs_cat, imp_cat, oof_cat, pred_cat = run_cv_cat(train_df, target, test_df, 4)


# In[49]:


plt.scatter(range(len(oof_cat)), oof_cat)


# In[50]:


# sub.columns
sub_cat = pd.DataFrame(data=[list(test_df_id), list(pred_cat.mean(axis=1))]).T#, columns=sub.columns)
sub_cat.columns = sub.columns
sub_cat.shot_id_number = sub_cat['shot_id_number'].astype(int)


# In[51]:


# os.makedirs('sub/')
sub_cat.to_csv('ankish_bansal_251094_prediction_4.csv', index=None)


# In[52]:



def train_xgb_model(X_train, y_train, X_valid, y_valid, features, param, X_test, 
                    num_round):
    """
    Args:
        X_train, X_valid: training and valid data
        y_train, y_valid: training and valid target
        X_test: test-data
        features: training features
    Return:
        oof-pred, test_preds, model, model_imp
    """
    _train = xgb.DMatrix(X_train[features], label=y_train, feature_names=list(features))
    _valid = xgb.DMatrix(X_valid[features], label=y_valid,feature_names=list(features))
    
    watchlist = [(_valid, 'valid')]
    clf = xgb.train(dtrain=_train, 
                    num_boost_round=num_round, 
                    evals=watchlist,
                    early_stopping_rounds=25, 
                    verbose_eval=200, 
                    params=param)
    
    valid_frame = xgb.DMatrix(X_valid[features],feature_names=list(features))
    oof  = clf.predict(valid_frame, ntree_limit=clf.best_ntree_limit)


    test_frame = xgb.DMatrix(X_test[features],feature_names=list(features))
    test_pred = clf.predict(test_frame, ntree_limit=clf.best_ntree_limit)

    
    xgb_imp = pd.DataFrame(data=[list(clf.get_fscore().keys()), 
                                 list(clf.get_fscore().values())]).T
    xgb_imp.columns = ['feature','imp']
    xgb_imp.imp = xgb_imp.imp.astype('float')
    
    return oof, test_pred, clf, xgb_imp


def run_cv_xgb(train_df, target, test_df, depth):

    features = train_df.columns
    params = {
        'eval_metric'     : 'auc',
        'seed'            : 1337,
        'eta'             : 0.01,
        'subsample'       : 0.7,
        'colsample_bytree': 0.5,
        'silent'          : 1,
        'nthread'         : 4,
        'Scale_pos_weight': 3.607,
        'objective'       : 'binary:logistic',
        'max_depth'       : depth,
        'alpha'           : 0.05
    }
    
    n_splits = 3
    random_seed = 1234
    feature_imp = pd.DataFrame()
    
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof_xgb = np.zeros(len(train_df))
    predictions = np.zeros((len(test_df),n_splits))
    clfs = []
##########################
    for fold_, (train_index, valid_index) in enumerate(folds.split(train_df, target)):
        print(train_index.shape, valid_index.shape)
        print("Fold {}".format(fold_))
    
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        X_train, X_valid = train_df.iloc[train_index,:], train_df.iloc[valid_index,:]
        features = X_train.columns
        

        num_rounds = 10000
        oof, test_pred, clf, xgb_imp = train_xgb_model(X_train, y_train, 
                                                       X_valid, y_valid, 
                                                       features, params, 
                                                       test_df, num_rounds)
        
        xgb_imp['fold'] = fold_
        feature_imp = pd.concat([feature_imp, xgb_imp], axis=0)
    
        oof_xgb[valid_index] = oof
        predictions[:,fold_] = test_pred
        clfs.append(clf)
        
        score = roc_auc_score(y_valid, oof)
        print( "  auc = ", score )
        print("="*60)
    
    feature_imp.imp = feature_imp.imp.astype('float')
    feature_imp = feature_imp.groupby(['feature'])['imp'].mean()
    feature_imp = pd.DataFrame(data=[feature_imp.index, feature_imp.values]).T
    feature_imp.columns=['feature','imp']
    feature_imp = feature_imp.sort_values(by='imp')


    return clfs, feature_imp, oof_xgb, predictions


# In[53]:


# clfs_xgb, imp_xgb, oof_xgb, pred_xgb = run_cv_xgb(train_df, target, test_df, 10)


# In[345]:


# pred_xgb.mean(axis=1)


# In[54]:


sub.head()


# In[55]:


sub.shape, pred_xgb.shape


# In[56]:


# pd.read_csv('sub/ankish_bansal_251094_prediction_4.csv').dtypes


# In[58]:


pd.read_csv('data/sample_submission.csv').dtypes

