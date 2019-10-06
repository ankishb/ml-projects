
To run this algorithms, please follow the instruction,

1. Place `final-submission-feature-engineering.ipynb` file in the current directory, where all data is placed in `data` directory, which looks like as:
For exp:
    os.listdir()
    	data/
    		train.csv
    	   test.csv
    	   sample_submission.csv
        final-submission-feature-engineering.ipynb
        final-submission-model-building.ipynb

2. Run `final-submission-feature-engineering.ipynb` file, it will save a `feature` file in the `data` directory, with name `train_test.csv`. The detail of feature engineering is explained in `my_approach.md`.
3. Run `final-submission-model-building.ipynb` file, it will build `2` model, one is `lighgbm` and other is `catboost` model. Final result is just the average of both prediction. This will save the `prediction` in `final_submission.csv` file in **current directory**.

