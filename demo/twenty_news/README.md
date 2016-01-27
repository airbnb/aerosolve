Twenty news demo
========================

# Introduction

In this demo we will train a multiclass model on the twenty news data set.
The data set we use was downloaded from https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/

## Pre-requisites

This demo assumes

  * Spark is installed http://spark.apache.org/downloads.html
  * spark-submit is in your path somewhere
  * Gradle is installed https://gradle.org/
  * Roughly 8 GB of free memory
  * python is installed somewhere

## Running the demo

Downloading the dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/
```
wget https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz
gzip -d 20_newsgroups.tar.gz
tar -xvf 20_newsgroups.tar
python convert_to_aerosolve.py 
```

This will convert the twenty news data set into one giant flat text file.

Descriptions of the dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups).

The headers have been scrubbed to make it more of an NLP task.
To see the data type

```
less 20_newsgroups.txt
```

The file format is label followed by a tab then the body of the post.

Save the dataset at `src/main/resources`, and make sure the path is consistent with the config file in `src/main/resources/income_prediction.conf`.

After you edit the config file, you can build the demo using:

`gradle shadowjar --info`

You can then see if the data was read properly using

`sh job_runner.sh DebugExample`

and also debug the transforms using

`sh job_runner.sh DebugTransform`

which should tokenize the sentences and delete the original sentence.

Then, make the training data for the models
```
sh job_runner.sh MakeTraining
```

Then you can train the model on the training data by running

`sh job_runner.sh TrainModel`

You can inspect the model using

```
spark-shell --master local[1] --jars build/libs/income_prediction-1.0.0-all.jar 
scala> import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.core.util.Util
scala> val model = sc.textFile("output/model/spline.model").map(Util.decodeModel)
scala> model.take(5).foreach(println)
ModelRecord(modelHeader:ModelHeader(modelType:spline, numRecords:53, numHidden:64, slope:1.0, offset:0.0))
ModelRecord(featureFamily:S, featureName:11th, weightVector:[-0.72265625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], minVal:1.0, maxVal:2.0)
ModelRecord(featureFamily:S, featureName:Local-gov, weightVector:[-0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], minVal:1.0, maxVal:2.0)
ModelRecord(featureFamily:S, featureName:Handlers-cleaners, weightVector:[-1.08203125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], minVal:1.0, maxVal:2.0)
ModelRecord(featureFamily:S, featureName:Male, weightVector:[0.2109375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], minVal:1.0, maxVal:2.0)
```

Next you can test the performance of the trained model on the training data and the testing data by running

`sh job_runner.sh EvalTraining`

For example, you may get:
```
 15/05/27 14:57:23 INFO IncomePredictionPipeline$: (!TRAIN_AUC,0.9144681939025233)
 15/05/27 14:57:23 INFO IncomePredictionPipeline$: (!TRAIN_AUC_MAX,0.9177016501516888)
 15/05/27 14:57:23 INFO IncomePredictionPipeline$: (!TRAIN_AUC_MIN,0.9090539660156829)
 15/05/27 14:57:23 INFO IncomePredictionPipeline$: (!TRAIN_AUC_STDDEV,0.00312018885799951)
 15/05/27 14:57:23 INFO IncomePredictionPipeline$: (!TRAIN_PREC@RECALL=0.000000,1.0)
 15/05/27 14:57:23 INFO IncomePredictionPipeline$: (!TRAIN_PREC@RECALL=0.205841,0.9853479853479854)
 15/05/27 14:57:23 INFO IncomePredictionPipeline$: (!TRAIN_PREC@RECALL=0.285295,0.9498938428874735)
 15/05/27 14:57:23 INFO IncomePredictionPipeline$: (!TRAIN_PREC@RECALL=0.356842,0.9055016181229774)
 15/05/27 14:57:23 INFO IncomePredictionPipeline$: (!TRAIN_PREC@RECALL=0.435404,0.8718079673135853)
 15/05/27 14:57:23 INFO IncomePredictionPipeline$: (!TRAIN_PREC@RECALL=0.511287,0.8252367229312474)
 15/05/27 14:57:23 INFO IncomePredictionPipeline$: (!TRAIN_PREC@RECALL=0.584237,0.7869781824428792)
 15/05/27 14:57:23 INFO IncomePredictionPipeline$: (!TRAIN_PREC@RECALL=0.656039,0.7404635094285303)
 15/05/27 14:57:23 INFO IncomePredictionPipeline$: (!TRAIN_PREC@RECALL=0.727586,0.6873493975903614)
 15/05/27 14:57:23 INFO IncomePredictionPipeline$: (!TRAIN_PREC@RECALL=0.815457,0.6242311822708191)
 15/05/27 14:57:23 INFO IncomePredictionPipeline$: (!TRAIN_PREC@RECALL=0.907792,0.5327046849274061)
 15/05/27 14:57:23 INFO IncomePredictionPipeline$: (!TRAIN_PREC@RECALL=1.000000,0.24081695331695332)
 15/05/27 14:57:23 INFO IncomePredictionPipeline$: (!TRAIN_PR_AUC,0.793211632913885)
```

`sh job_runner.sh EvalTesting`
For example, you may get:
```
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!BEST_THRESHOLD,0.2726899842637177)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_ACC,0.8522818008721823)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_AUC,0.915431622036853)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_AUC_MAX,0.9219664008396402)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_AUC_MIN,0.9119285048751522)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_AUC_STDDEV,0.003530404954202469)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_F1,0.6958391298849121)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_FPR,0.10534780860474467)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_PREC@RECALL=0.000000,1.0)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_PREC@RECALL=0.198388,0.9794608472400513)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_PREC@RECALL=0.270671,0.9463636363636364)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_PREC@RECALL=0.345034,0.9002713704206241)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_PREC@RECALL=0.421997,0.8619224641529474)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_PREC@RECALL=0.497660,0.8200514138817481)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_PREC@RECALL=0.569423,0.773851590106007)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_PREC@RECALL=0.641706,0.728238418412511)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_PREC@RECALL=0.715289,0.6774193548387096)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_PREC@RECALL=0.802652,0.6131082423038728)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_PREC@RECALL=0.902756,0.5232856066314996)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_PREC@RECALL=1.000000,0.23624078624078623)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_PRECISION,0.6774193548387096)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_PR_AUC,0.7794332286897141)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_RECALL,0.7152886115444618)
15/05/27 14:58:35 INFO IncomePredictionPipeline$: (!HOLD_RMSE,0.3165854782873086)
```

The results may vary slightly in different runs.
