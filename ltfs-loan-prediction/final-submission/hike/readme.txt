
To run this algorithms, use following instruction,

Place data file in directory part, which look like this
For exp:
    os.listdir('data')
    data/
    	train/
    		train.csv
    		user_features.csv
    	test.csv
    	sample_submission_only_headers.csv


1. Run `final-submission-feature-preperation.ipynb` file from this folder, This will generate lots of features, using graph network. Then the directory will have following more files.

	data/
		1_ranking_df.csv
		1_combine_all.csv
		1_katzing_11.csv
		1_katzing_22.csv
		50_u_vec.npy

2. Run `final-submission-training-part.ipynb`, this will run model and save final submission in data/ directory with name as `submission.csv`. The model will use all the available CPU. Please change the parameter `n_jobs`, if you want to limit the processes.

	data/
		submission.csv


Note: For my approach , you may want to look into `submission-detail.txt` file.