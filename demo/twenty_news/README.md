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

Save the dataset at `src/main/resources`, and make sure the path is consistent with the config file in `src/main/resources/twenty_news.conf`.

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
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@1,0.8617806197737334)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@10,0.9950811608460404)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@11,0.9960649286768323)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@12,0.9965568125922283)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@13,0.9975405804230202)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@14,0.9980324643384161)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@15,0.9990162321692081)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@16,0.9990162321692081)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@17,0.999508116084604)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@18,0.999508116084604)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@19,1.0)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@2,0.9513034923757994)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@20,1.0)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@3,0.9729463846532218)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@4,0.9803246433841614)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@5,0.985735366453517)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@6,0.9881947860304968)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@7,0.9901623216920806)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@8,0.9921298573536645)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (HOLD_PRECISION@9,0.9940973930152484)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_ALL_PAIRS_HINGE_LOSS,0.6203499147451094)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_MEAN_RECIPROCAL_RANK,0.960960504333188)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@1,0.9280227120908484)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@10,0.9993319973279893)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@11,0.999498997995992)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@12,0.9995546648853262)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@13,0.9995546648853262)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@14,0.9997773324426631)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@15,0.9997773324426631)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@16,0.9998886662213315)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@17,0.9999443331106658)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@18,0.9999443331106658)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@19,1.0)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@2,0.9867512803384547)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@20,1.0)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@3,0.9943776441772434)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@4,0.9964373190826097)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@5,0.9974949899799599)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@6,0.9981073257626364)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@7,0.9986083277666444)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@8,0.9988866622133156)
16/02/08 20:11:26 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@9,0.9991649966599866)
```

The results may vary slightly in different runs.
