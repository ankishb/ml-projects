

## Cross

The cross_validate function differs from cross_val_score in two ways -

    It allows specifying multiple metrics for evaluation.
    It returns a dict containing fit-times, score-times (and optionally training scores as well as fitted estimators) in addition to the test score.

For single metric evaluation, where the scoring parameter is a string, callable or None, the keys will be - ['test_score', 'fit_time', 'score_time']


```python
def cross_val_score(estimator, X, y=None, scoring=None, cv=None, n_jobs=1,
                    verbose=0, fit_params=None, pre_dispatch='2*n_jobs'):
    """Evaluate a score by cross-validation
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d.
    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.
    verbose : integer, optional
        The verbosity level.
    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    Returns
    -------
    scores : array of float, shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.
    """
```

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_val_score
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    >>> print(cross_val_score(lasso, X, y, cv=3))  # doctest: +ELLIPSIS
    [0.33150734 0.08022311 0.03531764]










def my_custom_loss_func(ground_truth, predictions):
    diff = np.abs(ground_truth - predictions).max()
    return np.log(1 + diff)

my_custom_scorer = make_scorer(my_custom_loss_func, greater_is_better=False)
grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=my_custom_scorer)




make_scorer takes as parameters:

    the function you want to use
    whether it is a score (greater_is_better=True) or a loss (greater_is_better=False),
    whether the function you provided takes predictions as input (needs_threshold=False) or needs confidence scores (needs_threshold=True)
    any additional parameters, such as beta in an f1_score.








```python
	Scoring 						Function 							Comment

Classification 	:

‘accuracy’ 						metrics.accuracy_score 	 
‘balanced_accuracy’ 			metrics.balanced_accuracy_score 	for binary targets
‘average_precision’ 			metrics.average_precision_score 	 
‘brier_score_loss’ 				metrics.brier_score_loss 	 
‘f1’ 							metrics.f1_score 					for binary targets
‘f1_micro’ 						metrics.f1_score 					micro-averaged
‘f1_macro’ 						metrics.f1_score 					macro-averaged
‘f1_weighted’ 					metrics.f1_score 					weighted average
‘f1_samples’ 					metrics.f1_score 					by multilabel sample
‘neg_log_loss’ 					metrics.log_loss 					requires predict_proba support
‘precision’ etc. 				metrics.precision_score 			suffixes apply as with ‘f1’
‘recall’ etc. 					metrics.recall_score 				suffixes apply as with ‘f1’
‘roc_auc’ 						metrics.roc_auc_score 


	 
Regression 	:

‘explained_variance’ 			metrics.explained_variance_score 	 
‘neg_mean_absolute_error’ 		metrics.mean_absolute_error 	 
‘neg_mean_squared_error’ 		metrics.mean_squared_error 	 
‘neg_mean_squared_log_error’ 	metrics.mean_squared_log_error 	 
‘neg_median_absolute_error’ 	metrics.median_absolute_error
```