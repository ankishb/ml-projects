## My approach:
1. Handling Null value
    + As there were only three feature with null value. i filled them with "unknown" string
2. Date based feature
    + i created [date, month, week, day, quater, weekday]
    + As we have to make prediction for the future observation(which means that the date of test dataset are not present in training dataset).
    + Also these feature didn't helped
3. Interaction based feature
    + these are superimportant
    + `R2` score improves with these feature
    ``` Here is the list of features, i used
    ['District']+['Postcode']
    ['District']+['Town']
    ['Town']+['Locality']
    ['Street1']+['Locality']
    ['Street2']+['Locality']
    ['AddressLine1']+['Street1']+['Street2']+['Locality']
    ['Street1']+['Locality']+['Town']
    ['Street2']+['Locality']+['Town']
    ['Street1']+['Street2']+['Locality']+['Town']
    ['District']+['Town']+['Locality']
    ['Street1']+['Street2']+['Locality']+['Town']+['District']+['Postcode']
    ['AddressLine1']+['Street2']+['Locality']+['Town']
    ```
4. Dummy feature
    + I used dummy feature for `['OldvNew', 'Duration', 'Price Category', 'Property Type']`. 
    + Here is a catch, i don't know, why my model doesn't find these good feature.
    + If we take practical approach, we clearly consider `'Property Type'` and `OldvNew` into account, but my models didn't find them interesting
    + The reason being is that the my `interaction` based feature had much higher weightage
5. Log-tranformation of target
    + To get a nice `normal-distribution`, i used `log` transformation on target
> Note: After all these feature enginering, i has 36 independent feature.
6. Model Building
    + I used `XGBoost` and `LightGBM` for final prediction. There parameters details can be found in `final-submission.ipynb` file.
    + Cross validation is performed with split size of `2`. As i noticed that dataset is quite big and the random sampling appraoch gives me approximately same mean average value. I go with split-size of `2`.
    + 
7. Bayesian optimization is performed to find the good parameter.
    + unsucessful to optimize `xgboost`, due to time cosntrained
    + successfully tuned paarmeters of `lighgbm`, which improves the `R2-score` from `54 to 63`
8. Meta feature building
    + My final submission is based on the stackign of `5` models.
    + `out of fold` prediction is used as meta feature for final model

## Unsuccessful attempt:
- Linear model disn't work, i could not find the reason behind. It also take my `45 minutes`


Note: Please use `exponential` operation on final result, if need for finall prediction. On the behalf of my observation on platform's result, my final result is `log` transformed.