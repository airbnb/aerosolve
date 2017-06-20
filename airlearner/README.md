Airlearner
=========

What is it?
-----------
A practical machine learning library designed for production work. Current components:
* [Binary Regression](https://github.com/airbnb/aerosolve/blob/master/airlearner/airlearner-strategy/src/main/scala/com/airbnb/common/ml/strategy/trainer/BinaryRegressionTrainer.scala) 
* [XGBoost Pipeline](https://github.com/airbnb/aerosolve/tree/master/airlearner/airlearner-xgboost)

[Binary Regression](https://github.com/airbnb/aerosolve/blob/master/airlearner/airlearner-strategy/src/main/scala/com/airbnb/common/ml/strategy/trainer/BinaryRegressionTrainer.scala) 
-------------------
Binary regression refers to regression problem with both boolean and float label. 
Pricing problem is a typical binary regression problem, such that true label associated with the sold price,
 while false label associated with unsold price. 
Traditional regression can't deal with this type of problem without bias, 
because either drop all false label samples or keep it with unsold price are biased.
Binary regression solve this type of problem.


[XGBoost Pipeline](https://github.com/airbnb/aerosolve/tree/master/airlearner/airlearner-xgboost)
-------------------

XGBoost Pipeline productionizes xgboost in spark + HDFS/Hive environment. It supports

* Transform hive data into xgboost training data
* Training, evaluation and scoring pipeline
* MonteCarlo param search and save param search in hive table
* Save Model and model output into HDFS and Hive.
