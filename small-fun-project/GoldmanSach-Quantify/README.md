
## My approach:
1. Data Cleaning
	+ removed punctuation
	+ also removed keyword like `len`, `time`
	+ append all the `info` in a sentence
2. feature generation
	+ use regex for further cleaning 
	+ generate `TFIDF` based feature
	+ removed highly occurred words such as status and other
	+ used `htttps-links` as a string
3. Clustering
	+ kmean-clustering is used
	+ topic modelling)(but didn't worked out properly, need more tuning and cleaning)
