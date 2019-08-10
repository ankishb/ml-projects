
To run this algorithms, please use following instruction,

Place `final-submission.ipynb` file in `LTFS-loan-prediction` directory part, which look like this.
For exp:
    os.listdir('LTFS-loan-prediction')
    LTFS-loan-prediction/
    	train_aox2Jxw/
    		train.csv
    	test_bqCt9Pv.csv
    	sample_submission_24jSKY6.csv
    	final-submission.ipynb

1. Run `final-submission.ipynb` file from this folder, This will start with features preprocessing and engineering and then fit a gbm model. It will save submission file with name `final-submission.csv` in the same('data') folder.


2. For my approach, you may want to read ipynb notebook code, which i have put small comment. It is very simple, just feature engineering and final lightgbm model on those features.