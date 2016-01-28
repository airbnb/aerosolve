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

Next you can test the performance of the trained model on the training data and the testing data by running

`sh job_runner.sh EvalModel`

You should get a bunch of metrics like this:

```
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_ALL_PAIRS_HINGE_LOSS,15.90144201708876)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_MEAN_RECIPROCAL_RANK,0.7062573465876217)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@1,0.6123954746679784)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@10,0.8563698967043778)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@11,0.8642400393507133)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@12,0.8726020659124447)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@13,0.8804722085587802)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@14,0.8893261190359075)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@15,0.8962124938514511)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@16,0.9045745204131825)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@17,0.9129365469749139)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@18,0.9222823413674373)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@19,0.9798327594687654)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@2,0.720609936055091)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@20,1.0)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@3,0.7624200688637481)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@4,0.7904574520413182)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@5,0.8106246925725529)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@6,0.8214461387112642)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@7,0.8337432365961633)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@8,0.8411214953271028)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (HOLD_PRECISION@9,0.8489916379734382)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_ALL_PAIRS_HINGE_LOSS,13.046479317292844)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_MEAN_RECIPROCAL_RANK,0.7604834563965203)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@1,0.6835894010242708)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@10,0.8811511912714317)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@11,0.88666221331552)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@12,0.8937319082609664)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@13,0.9004119349810733)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@14,0.9068692941438432)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@15,0.9133266533066132)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@16,0.9200066800267201)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@17,0.9281340458695169)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@18,0.9364284123803162)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@19,0.9893676241371632)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@2,0.7759964373190826)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@20,1.0)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@3,0.8075038966822534)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@4,0.8268203072812291)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@5,0.8409040302827878)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@6,0.8517034068136272)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@7,0.8604431084391004)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@8,0.8675684702738811)
16/01/27 16:57:53 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@9,0.8743041638833222)
```

The results may vary slightly in different runs.
