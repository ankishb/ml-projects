
'UniqueID',
'disbursed_amount',   == corrected (outliers removed)
'asset_cost',         == corrected (outliers removed)
# 'ltv',                == 'ltv_round_cat' == (very dangerous features)
'supplier_id',        == (will Drop it)
'Current_pincode_ID', == (will Drop it)
'Employment.Type',    == corrected
'State_ID',           == use for grouping
'Employee_code_ID',   == use for grouping
'manufacturer_id',    == use for grouping
'branch_id',          == use for grouping (Use target-encoding)
'MobileNo_Avl_Flag',  == dropped
'Aadhar_flag',        ==
'PAN_flag',           ==
'VoterID_flag',       == dropped
'Driving_flag',       == dropped 
'Passport_flag',      == dropped
# 'PRI.NO.OF.ACCTS',    == no_of_loans
'PRI.ACTIVE.ACCTS',   == no_of_acc
'PRI.OVERDUE.ACCTS',  == no_of_acc_overdue
# 'PRI.CURRENT.BALANCE',== negative_income == NEED_HELP(SUSPICIOUS)

'PRI.SANCTIONED.AMOUNT', == Removed Outliers
'PRI.DISBURSED.AMOUNT',  == Removed Outliers

From 'PRI.SANCTIONED.AMOUNT' & 'PRI.DISBURSED.AMOUNT' == 'obtained_amount'
 

'SEC.NO.OF.ACCTS',       == dropped
'SEC.ACTIVE.ACCTS',      == dropped
'SEC.OVERDUE.ACCTS',     == dropped
# 'SEC.CURRENT.BALANCE',   == NEED_HELP(SUSPICIOUS)
'SEC.SANCTIONED.AMOUNT', == corrected
'SEC.DISBURSED.AMOUNT',  == corrected
'PRIMARY.INSTAL.AMT',    == Removed Outliers #Difficult to handle (removed outliers but still distributions is not good)
'SEC.INSTAL.AMT',        == dropped
'NO.OF_INQUIRIES',       == corrected
'loan_default',          == 
'date_of_birth'          == 'Age(in years)', 'Age(in month)' 
'credit_hist_year',      == corrected
'credit_hist_month',     == 
'credit_hist__total_month'==
'loan_tenure_year',      == corrected
'loan_tenure_month',     ==
'loan_tenure_total_month'==
'day_of_disbursal',      == 
'week_of_disbursal'      == 
'month_of_disbursal',    == 
'year_of_disbursal',     == dropped
'Bureau_desc',           == 
'bureau_score'           == corrected

'NEW.ACCTS.IN.LAST.SIX.MONTHS',        == corrected
'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS', == corrected

From 'credit_hist_year', &  'credit_hist_month', == 'credit_hist__total_month'
From 'loan_tenure_year', & 'loan_tenure_month', == 'loan_tenure_total_month'












# time-series model
# def run_lgb_small(file_path, train_df, target, test_df, test_ids, sub, depth):
def run_lgb_small(train_df, target, leaves=None):
    
    splits = 4
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb

    features = train_df.columns
    random_seed = 2019
    feature_imp = pd.DataFrame()
    
    param = {
        'bagging_freq'           : 5,
        'bagging_fraction'       : 0.33,
        'boost_from_average'     : 'false',
        'boost'                  : 'gbdt',
        'feature_fraction'       : 0.3,
        'learning_rate'          : 0.01,
        'max_depth'              : -1,
        'metric'                 : 'auc',
#         'min_data_in_leaf'       : 100,
#         'min_sum_hessian_in_leaf': 10.0,
        'num_leaves'             : 30,
        'num_threads'            : 4,
        'tree_learner'           : 'serial',
        'objective'              : 'binary',
        'verbosity'              : 1,
    #     'lambda_l1'              : 0.001,
        'lambda_l2'              : 0.01
    }   
    if leaves is not None:
        param['num_leaves'] = leaves
        print("using leaves: ", param['num_leaves'])

    n_split = splits
    num_round = 10000

    valid_splits = int(train_df.shape[0]/splits)
    indexes = train_df[['disbursal_week', 'disbursal_day']].sort_values(
        by=['disbursal_week', 'disbursal_day'])
    
    train_index = indexes[:-valid_splits].index
    valid_index = indexes[-valid_splits:].index
        
    y_test_pred = 0


    idx = 0
    y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
    X_train, X_valid = train_df.iloc[train_index,:], train_df.iloc[valid_index,:]
    
    X_train.drop(['disbursal_week','disbursal_day'], axis=1, inplace=True)
    X_valid.drop(['disbursal_week','disbursal_day'], axis=1, inplace=True)
    
    features = X_train.columns

    oof, test_pred, clf, lgb_imp = train_lgb_model(X_train, y_train, 
                                                   X_valid, y_valid, 
                                                   features, param, 
                                                   X_test)
    lgb_imp['fold'] = idx
    feature_imp = pd.concat([feature_imp, lgb_imp], axis=0)
    
#     _train = lgb.Dataset(X_train[features], label=y_train,
#                           feature_name=list(features))
#     _valid = lgb.Dataset(X_valid[features], label=y_valid,
#                           feature_name=list(features))

#     clf = lgb.train(param, _train, num_round, valid_sets = [_train, _valid], 
#                     verbose_eval=200, early_stopping_rounds = 25)

                         
#     pred = clf.predict(X_valid[features], num_iteration=clf.best_iteration)
    
    score = roc_auc_score(y_valid, oof)
    print( "  auc = ", score )
    print("="*60)

#     y_test_pred = fit_model.predict_proba(test_df)[:,1]

#     sub_df = pd.DataFrame({"ID_code":test_ids})
#     sub_df["target"] = y_test_pred
#     sub_df.columns = sub.columns

#     sub_df.to_csv('submission/catboost_{}.csv'.format(file_path), index=None)

    lgb_imp = pd.DataFrame(data=[clf.feature_name(), list(clf.feature_importance())]).T
    lgb_imp.columns = ['feature','imp']
    lgb_imp = lgb_imp.sort_values(by='imp', ascending=False)
    plt.figure(figsize=(12,15))
    plt.barh(lgb_imp.feature, lgb_imp.imp)
    plt.show()
    
    return clf, lgb_imp, X_valid, y_valid