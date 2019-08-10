very-intersting ==> https://www.kaggle.com/xaviermaxime/light-gbm-with-simple-engineered-features



https://www.dummies.com/programming/big-data/data-science/data-science-how-to-create-interactions-between-variables-with-python/

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
regression = LinearRegression(normalize=True)
crossvalidation = KFold(n=X.shape[0], n_folds=10, shuffle=True, random_state=1)


df = pd.DataFrame(X,columns=boston.feature_names)
baseline = np.mean(cross_val_score(regression, df, y, scoring=‘r2’, cv=crossvalidation,
n_jobs=1))
interactions = list()
for feature_A in boston.feature_names:
for feature_B in boston.feature_names:
if feature_A > feature_B:
df[‘interaction’] = df[feature_A] * df[feature_B]
score = np.mean(cross_val_score(regression, df, y, scoring=‘r2’,
cv=crossvalidation, n_jobs=1))
if score > baseline:
interactions.append((feature_A, feature_B, round(score,3)))
print ‘Baseline R2: %.3f’ % baseline
print ‘Top 10 interactions: %s’ % sorted(interactions, key=lambda(x):x[2],
reverse=True)[:10]
Baseline R2: 0.699
Top 10 interactions: [(‘RM’, ‘LSTAT’, 0.782), (‘TAX’, ‘RM’, 0.766),
(‘RM’, ‘RAD’, 0.759), (‘RM’, ‘PTRATIO’, 0.75),
(‘RM’, ‘INDUS’, 0.748), (‘RM’, ‘NOX’, 0.733),
(‘RM’, ‘B’, 0.731), (‘RM’, ‘AGE’, 0.727),
(‘RM’, ‘DIS’, 0.722), (‘ZN’, ‘RM’, 0.716)]


polyX = pd.DataFrame(X,columns=boston.feature_names)
baseline = np.mean(cross_val_score(regression, polyX, y,
scoring=‘mean_squared_error’,
cv=crossvalidation, n_jobs=1))
improvements = [baseline]
for feature_A in boston.feature_names:
polyX[feature_A+’^2’] = polyX[feature_A]**2
improvements.append(np.mean(cross_val_score(regression, polyX, y,
scoring=‘mean_squared_error’, cv=crossvalidation, n_jobs=1)))
for feature_B in boston.feature_names:
if feature_A > feature_B:
polyX[feature_A+’*’+feature_B] = polyX[feature_A] * polyX[feature_B]
improvements.append(np.mean(cross_val_score(regression, polyX, y,
scoring=‘mean_squared_error’, cv=crossvalidation, n_jobs=1)))





1. Try out simple multiplication of cat column with data type as str

1. 1 1
2. 2 1
3. 1 2
4. 1 2
5. 2 2

Using the upper technique, new feature will be

1. 1 1 1:1
2. 2 1 2:1
3. 1 2 1:2
4. 1 2 1:2
5. 2 2 2:2



==> Apply sigmoid/tanh function for tranformation
==> Binary variable, whether feature has null or not
==> check if is is fututre dataset in test
==> For faster experiment, use sampling method to select subset of training data, but don't touch validation
==> Always explore model/feature-engineering by subsetting the whole data

==> min_sample_leaf help in better generalization, choose 1,3,5,10,25 for data-set of range 100,000 sample, for bigger data-set tune this parameter to 100,1000 etc.
==> Choose max_feature to 0.5,sqrt,log for better generalization.

==> Check out distribution of danger column w.r.t each label 0/1



https://www.youtube.com/watch?v=42Oo8TOl85I
https://github.com/h2oai/h2o-tutorials/tree/master/h2o-world-2017/automl
H20 ==> feature-engineering (https://github.com/h2oai/h2o-tutorials/blob/78c3766741e8cbbbd8db04d54b1e34f678b85310/best-practices/feature-engineering/feature_engineering.ipynb)

https://github.com/h2oai/h2o-3/blob/master/h2o-py/h2o/targetencoder.py




The issue is because you are trying encoding multiple categorical features. I think that is a bug of H2O, but you can solve putting the transformer in a for loop that iterate over all categorical names.

import numpy as np
import pandas as pd
import h2o
from h2o.targetencoder import TargetEncoder
h2o.init()

df = pd.DataFrame({
'x_0': ['a'] * 5 + ['b'] * 5,
'x_1': ['c'] * 9 + ['d'] * 1,
'x_2': ['a'] * 3 + ['b'] * 7,
'y_0': [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
})

hf = h2o.H2OFrame(df)
hf['cv_fold_te'] = hf.kfold_column(n_folds=2, seed=54321)
hf['y_0'] = hf['y_0'].asfactor()
cat_features = ['x_0', 'x_1', 'x_2']

for item in cat_features:
target_encoder = TargetEncoder(x=[item], y='y_0', fold_column = 'cv_fold_te')
target_encoder.fit(hf)
hf = target_encoder.transform(frame=hf, holdout_type='kfold',
seed=54321, noise=0.0)
hf














########## Handle Outliers ##########

For linear model, outlier should be removed, like following

pd.Series(x).hist(bins=30)
lower_bound, upper_bound = np.percentile(1,99)
y = np.clip(X, lower_bound, upper_bound)
pd.Series(y).hist(bins=30)

Preprocessing: Rank method
linear_model, knn, NN

scipy.stats.rankdata

np.log(1+x)

Raising to the power < 1
np.sqrt(x+2/3)

fractional_part
2.99 ==> 0.99
1.49 ==> 0.49


LabelEncoder ==> Alphabatical encoding
[S,C,Q] ==> [2,1,3]

pandas.factorize ==> order of appearance
[S,C,Q] ==> [1,2,3]

Frequency-encoding: frequency of category is correlated with target value, linear model will utilize this dependency.
It also helpful for tree based models.

encoding = titanic.groupby('Embarked').size()
encoding = encoding/len(titanic)
titanic['enc'] = titanic.Embarked.map(encoding)


If we have same frequency for some catgories, then it will not be distinguishable, 
apply ranking method on that.
scipy.stats.rankdata


one-hot-encoding : for knn/nn, non-tree based model
pandas.get_dummies
sklearn.preprocessing.OneHotEncoder
Try using sparse matrix for highly cardinal data, 
xgboost, lightgbm and catboost can handle them easily.

Interaction of cat-features ==> 
can help linear models and KNN





Date and time:
1. periodicity: Day number in week, month, season, year, second, minute, hour.
2. time since
	- no of days left untill next holidays
	- time passed after last holiday

3. Difference between dates
	- date_feat2 - date_feat1



Handle Missing Values:
1. IsNull feature 
2. Missing-values reconstruction in time series data-set using nearest neighbour approach.
3. Avoid filling nan before feature generation 
4. replace outlier with missing values can be helpful sometimes.
5. xgboost can handle missing values

