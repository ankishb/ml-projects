
I have briefly explained my approach:

## Technical details:
- Ubuntu 14
- RAM 8GB
- CPU I5


# Data Cleaning and feature engineering:
- date-time based feature
- Aggregation based feature on following columns:
    + memberid
    + resort_id
    + cluster_code
    + resort_region_code
    + resort_type_code
    + state_code_resort
    + state_code_residence
    + On Date
- day-diff between `checkout` and `checkin`

# Model and its parameters:
- **LightGBM**
    'max_bins'        : 63,
    'learning_rate'   : 0.01,
    'num_threads'     : 4,
    'metric'          : 'rmse',
    'boost'           : 'gbdt',
    'tree_learner'    : 'serial',
    'objective'       : 'root_mean_squared_error',
    'verbosity'       : 1,
    'b_frac'          : 0.36,
    'data_in_leaf'    : 495.4,
    'f_frac'          : 0.5,
    'hessian'         : 5.04,
    'l1'              : 1.06,
    'l2'              : 2.5,
    'leaves'          : 89.6,
    'split_gain'      : 0.2
- **CatBoost**
    'bagging_temperature'  : 0.08,
    'depth'                : 7,
    'l2_leaf_reg'          : 3.4,
    'random_strength'      : 1.12,
    'border_count'          : 63,
    'early_stopping_rounds' : 50,
    'random_seed'           : 1337,
    'task_type'             : 'CPU', 
    'loss_function'         : "RMSE", 
    'iterations'            : 10000, 
    'learning_rate'         : 0.01,

# Final Submission
- Average of both model's prediction
