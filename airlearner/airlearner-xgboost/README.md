How to use XGBoost Pipeline
=========

XGBoost Pipeline helps use xgboost with spark + hdfs/hive evnviroment. It supports
 * Transform hive data into xgboost training data
 * Training, evaluation and scoring pipeline
 * MonteCarlo param search and save param search in hive table
 * Save Model and model output into hdfs and hive.

Training steps:
* [Create Hive Tables](https://github.com/airbnb/aerosolve/blob/master/airlearner/xgboost/hive/setup.hql) 
* Preparing Training/Eval data in Hive. 
* Prepare xgboost_sample table, decide sample rate for each model id.
* Create concrete TrainingData From [TrainingModelData](https://github.com/airbnb/aerosolve/blob/master/airlearner/xgboost/src/main/scala/com/airbnb/common/ml/xgboost/data/TrainingModelData.scala)
  if your training data in hive follows `label, node_id, features,...` use BaseTrainingModelData.
* Create conf file from[search_template](https://github.com/airbnb/aerosolve/blob/master/airlearner/strategy/src/main/resources/search_template.conf)    
* `gradlew shadowJar` build jar
* use [submit.sh](https://github.com/airbnb/aerosolve/blob/master/airlearner/strategy/src/bash/submit.sh) `./submit.sh task_name config_name > log_name 2>&1`


