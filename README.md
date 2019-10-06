# ml-projects
This reposatry contains my DS and ML-contest's projects, along with my personal fun project. I have dealt with diverse set of problem/data/metrics. Following is the summary of each project, which contains the `type of dataset`, `type of problem` and `my-approach` to handle that(all in very brief). More details can be found in each subdirectory.

### Flipkart Object Detection
    - DataSet:
        + Image
    - Type:
        + Bounding Box prediction
    - My Approach:
        + Designed a visual feature pipeline with attention on the object in image
        + Data Augmentation Technique along with its bounding box
        + Used `Single Stage Detector` Approach
        + Focal Loss with `YOLO` and `SSD`


### Amazon Product Review classification
    - DataSet:
        + Text
    - Type:
        + Classification
    - My Approach:
        + Data Cleaning/feature enginnering
        + Linear/Non-Linear Model
        + Deep Learning Attention based Model
        + Pretrained Bert Model
        + Ensemble


### HDFC Risk Prediction
    - DataSet:
        + `2500` unknown predictors
    - Type:
        + Classification 
    - My Approach:
        + Feature Understanding(`EDA`)
        + feature engineering
        + designed feature interaction tools
        + ensemble model using `xgboost/lighgbm/catboost` and `linear/non-linear` simple model
        + statistical model to understand the feature importance using `p-values`


### Hike Friend Recommendation
    - DataSet:
        + Relation Feature
        + Very big Dataset(45M observation)
    - Type:
        + Link Prediction
    - My Approach:
        + Graph Based features such as (`adamic-adar`, `common-resource-allocation`,...)
        + `SVD` feature for each user
        + `Comunity-clustering`
        + `Subsemble`(I did this after competition is over, to understand more about sampling and model building)
        + `neighbour-based` feature(Removed highly cardinal feature)
        + Also tried `Deep learning approach` (Graph Embedding), but couldn't handle at that time properly


### Club Mahindra Hotel Room Price Prediction
    - DataSet:
        + Category
        + Numerical
        + Relational Dataset
    - Type:
        + Regression
        + Hotel-Room Price Prediction 
    - My Approach:
        + Feature engineering
            1. date-time based feature
            2. Aggregation based feature
            3. Relational Features
        + Ensemble using different set of `tranformed` target space


### Cifar-10 Classification using Conditional Feature
    - DataSet:
        + Image
    - Type:
        + Classification
        + Comparison between ResNet and my modified feature pipeline
    - My Approach:
        + Developed a weighted feature pipeline using global and local feature.
        + Global feature put constrained on local feature, to specifically focused on features of object in image
        + Better attention map around object, which reflect its learned feature.
        + Improved score by `1.37%` over `Resnet`


### Facenet 
    - DataSet:
        + Image
    - Type:
        + Face Verification
    - My Approach:
        + Matching Network Approach
        + Built a Student-Attentdance hardware using arduino
        + Hard Mining Approach(generate all permutation between classes to handle small dataset)
        + network-in-network approach to handle overfitting as i have very small dataset.
        + Achieved `93%` accuracy


### Few Shot Learning(Prototype Network)
    - DataSet:
        + Image
    - Type:
        + Classification using small set of data
    - My Approach:
        + Prototype Algorithm implementation
        + There is more to this(will update in future)


### JP.Morgan House Price Prediction
    - DataSet:
        + Category
        + Numerical
    - Type:
        + Regression (House prices prediction)
    - My Approach:
        + Date based feature and Dummy feature
        + Interaction based feature 
        + Bayesian optimization
        + `out of fold prediction` to generate `Meta feature` for `ensemble`


### Hackerearth Platform Recommendation System
    - DataSet:
        + Text
    - Type:
        + User-Problem Rating Prediction
    - My Approach:
        + My main concerns was to handle following question carefully:
        1. What is the strongest and weakest area of user?
        2. What is the level of problem?
        3. What problem user have just solved?
        4. If user gets stuck at current problem, what problem should help him(to gain confidence and to improve skill in that area)?
        5. Exploration and explotation strategy in recommending problem
        6. And many more?


### LTFS Loan Status prediction
    - DataSet:
        + Category
        + Numerical
    - Type:
        + Classification
    - My Approach:
        + 


### Segmentation
    - DataSet:
        + Image
    - Type:
        + Segmentation
    - My Approach:
        + 


### Future sale Prediction
    - DataSet:
        + Relational feature
        + Time Feature
        + Categorical
        + Numerical
    - Type:
        + Future Sales Prediction for different store in different cities
    - My Approach:
        + 


### Gartner Retention Status Prediction
    - DataSet:
        + Image
    - Type:
        + Classification 
    - My Approach:
        + EDA
        + Feature Engineering


### Stock Prediction
    - DataSet:
        + Time Series stock prices
    - Type:
        + Future price prediction
        + Regression
    - My Approach:
        + Deep learning approach using RNN and LSTM

