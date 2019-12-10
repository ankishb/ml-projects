# ml-projects
This reposatry contains my DS and ML-contest's projects, along with my personal fun project. I have dealt with diverse set of problem/data/metrics. Following is the summary of each project, which contains the `type of dataset`, `type of problem` and `my-approach` to handle that(all in very brief). More details can be found in each subdirectory.

> If you want to look the following text in a table format, [click here](https://github.com/ankishb/ml-projects/blob/master/README2.md)
 

### [Flipkart Object Detection](https://github.com/ankishb/ml-projects/tree/master/conditional-object-detection)
- DataSet:
    + Image
- Objective:
    + Bounding Box prediction
- My Approach:
    + Designed a visual feature pipeline with attention on the object in image
    + Data Augmentation Technique along with its bounding box
    + Used `Single Stage Detector` Approach
    + Focal Loss with `YOLO` and `SSD`


### [Amazon Product Review classification](https://github.com/ankishb/ml-projects/tree/master/amazon-ml)
- DataSet:
    + Text
- Objective:
    + Classification
- My Approach:
    + Data Cleaning/feature enginnering
    + Linear/Non-Linear Model
    + `Deep Learning Attention Model`
    + `Pretrained Bert Model`
    + Ensemble


### [HDFC Risk Prediction](https://github.com/ankishb/ml-projects/tree/master/hdfc-ml)
- DataSet:
    + `2500` unknown predictors
- Objective:
    + Classification 
- My Approach:
    + Feature Understanding(`EDA`)
    + feature engineering
    + designed feature interaction tools
    + ensemble model using `xgboost/lighgbm/catboost` and `linear/non-linear` simple model
    + statistical model to understand the feature importance using `p-values`


### [Hike Friend Recommendation](https://github.com/ankishb/ml-projects/tree/master/hike-friend-recommendation)
- DataSet:
    + Very big Dataset(45M observation, graph edge-representation)
    + Relational Feature
    + Category + Numerical
- Objective:
    + Link Prediction
- My Approach:
    + Graph Based features such as (`adamic-adar`, `common-resource-allocation`,...)
    + `SVD` feature for each user
    + `Comunity-clustering`
    + `Subsemble`(I did this after competition is over, to understand more about sampling and model building)
    + `neighbour-based` feature(Removed highly cardinal feature)
    + Also tried `Deep learning approach` (Graph Embedding), but couldn't handle at that time properly


### [Club Mahindra Hotel Room Price Prediction](https://github.com/ankishb/ml-projects/tree/master/club-mahindra)
- DataSet:
    + Category + Numerical
    + Relational Dataset
- Objective:
    + Regression
- My Approach:
    + Feature engineering
        1. `date-time` based feature
        2. `Aggregation` based feature
        3. `Relational` Features
    + Ensemble using different set of `tranformed` target space


### [Cifar-10 Classification using Conditional Feature](https://github.com/ankishb/ml-projects/tree/master/cifar-10-resnet)
- DataSet:
    + Image
- Objective:
    + Comparison between ResNet and my modified feature pipeline
    + Classification
- My Approach:
    + Developed a `weighted feature pipeline using global and local feature`.
    + `Global feature put constrained on local feature, to specifically focused on features of object` in image
    + `Better attention map around object`, which reflect its learned feature.
    + Improved score by `1.37%` over `Resnet`


### [Facenet](https://github.com/ankishb/ml-projects/tree/master/facenet)
- DataSet:
    + Image
- Objective:
    + Face Verification
- My Approach:
    + `Matching Network Approach`
    + Build a `Student-Attentdance hardware using arduino`
    + `Hard Mining Approach`(generate all permutation between classes to handle small dataset)
    + `network-in-network` approach to handle overfitting as i have very small dataset.
    + Achieved `93%` accuracy


### [Few Shot Learning(Prototype Network)](https://github.com/ankishb/ml-projects/tree/master/few-shot-classification)
- DataSet:
    + Image
- Objective:
    + Classification (training on very small dataset)
- My Approach:
    + `Prototype Algorithm` implementation
    + There is more to this(will update in future)


### [JP.Morgan House Price Prediction](https://github.com/ankishb/ml-projects/tree/master/jp-morgan)
- DataSet:
    + Category + Numerical
- Objective:
    + Regression 
- My Approach:
    + Date based feature and Dummy feature
    + `Interaction based feature` 
    + `Bayesian optimization`
    + `out of fold prediction` to generate `Meta feature` for `ensemble`


### [Hackerearth Platform Recommendation System](https://github.com/ankishb/ml-projects/tree/master/recommendation-system)
- DataSet:
    + Text
- Objective:
    + User-Problem Rating Prediction
- My Approach:
    + My main concerns was to handle following question carefully:
        1. What is the strongest and weakest area of user?
        2. What is the level of problem?
        3. What problem user have just solved?
        4. If user gets stuck at current problem, what problem should help him(to gain confidence and to improve skill in that area)?
        5. Exploration and explotation strategy in recommending problem
        6. And many more?


### [LTFS Loan Status prediction](https://github.com/ankishb/ml-projects/tree/master/ltfs-loan-prediction)
- DataSet:
    + Category + Numerical
- Objective:
    + Classification
- My Approach:
    + 


### [Segmentation](https://github.com/ankishb/ml-projects/tree/master/segmentation)
- DataSet:
    + Image
- Objective:
    + Segmentation
- My Approach:
    + Implemented an U-Net architecture on blood cell Dataset.
    + fully convolutional network on traffic-street dataset.
    + Finally experimented with generative adverserial network for better generalization in the presence of limited dataset.


### [Future sale Prediction](https://github.com/ankishb/ml-projects/tree/master/small-fun-project/future-sale-pred)
- DataSet:
    + Relational feature
    + Time-Series Feature
    + Categorical + Numerical
- Objective:
    + Future Sales Prediction for different store in different cities
- My Approach:
    + 


### [Gartner Retention Status Prediction](https://github.com/ankishb/ml-projects/tree/master/small-fun-project/gartner)
- DataSet:
    + Image
- Objective:
    + Classification 
- My Approach:
    + EDA
    + Feature Engineering


### [Stock Prediction](https://github.com/ankishb/ml-projects/tree/master/small-fun-project/collect-imp-tensor-spyder/time-series-prediction)
- DataSet:
    + Time-Series stock prices
- Objective:
    + Future price prediction
    + Regression
- My Approach:
    + Deep learning approach using RNN and LSTM

