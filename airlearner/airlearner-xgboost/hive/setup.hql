-- define following params and run this file before using xgboost pipeline
-- tables are partitioned by date(ds) and model name
-- model partition is used in development phase to try different version of models
-- id is the model id, i.e. use kd tree as ID, each kd tree has its own model.

{%- set hdfs_base_location = 'hdfs://company/project/' %}

use xgboost;

-- param search for models
-- TODO replace external table to managed table with ORC format.
CREATE EXTERNAL TABLE IF NOT EXISTS grid_search (
  -- use id for different params set in one model
  id          INT
  , params    ARRAY<DOUBLE>
  , result    DOUBLE
) PARTITIONED BY (ds STRING, model STRING)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
COLLECTION ITEMS TERMINATED BY '\001'
STORED AS TEXTFILE
LOCATION '{{hdfs_base_location}}grid_search'
;

--AUC for xgboost model
CREATE EXTERNAL TABLE IF NOT EXISTS xgboost_auc (
  id                    bigint
  , auc                 double
  , totalPositiveCount  double
  , totalNegativeCount  double
) PARTITIONED BY (ds string, model STRING)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION '{{hdfs_base_location}}xgboost_auc'
;

CREATE EXTERNAL TABLE IF NOT EXISTS feature_importance (
  id                    STRING
  , feature             STRING
  , score               bigint
) PARTITIONED BY (ds string, model STRING)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION '{{hdfs_base_location}}feature_importance'
;

--xgboost model output
CREATE EXTERNAL TABLE IF NOT EXISTS xgboost_score (
  id            BIGINT
  , base_value  DOUBLE
  , score       DOUBLE
) PARTITIONED BY (ds STRING)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION '{{hdfs_base_location}}score'
;

-- store sample rate of diffent xgboost models.
CREATE TABLE xgboost_sample AS SELECT
  id                BIGINT
  , total           BIGINT -- total number of training samples
  , sample_rate     DOUBLE
) PARTITIONED BY (model STRING)
;
