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
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_ALL_PAIRS_HINGE_LOSS,15.741044936394726)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_MEAN_RECIPROCAL_RANK,0.8479250372105728)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@1,0.7663551401869159)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@10,0.9822921790457452)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@11,0.984751598622725)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@12,0.9862272503689129)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@13,0.9891785538612887)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@14,0.9906542056074766)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@15,0.9945892769306444)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@16,0.9955730447614363)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@17,0.9965568125922283)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@18,0.9975405804230202)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@19,0.9980324643384161)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@2,0.8691588785046729)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@20,1.0)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@3,0.9163797343826857)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@4,0.9458927693064437)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@5,0.9576979832759469)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@6,0.9660600098376783)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@7,0.9719626168224299)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@8,0.9773733398917855)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (HOLD_PRECISION@9,0.9813084112149533)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_ALL_PAIRS_HINGE_LOSS,13.321086623122758)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_MEAN_RECIPROCAL_RANK,0.8711004893121844)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@1,0.7963148519260743)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@10,0.9873636161211311)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@11,0.9884769539078156)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@12,0.9899242930305054)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@13,0.9915386328211979)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@14,0.992540636829214)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@15,0.9935983077265642)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@16,0.9946559786239145)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@17,0.9957136495212647)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@18,0.9966599866399466)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@19,0.997773324426631)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@2,0.8991872634157203)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@20,1.0)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@3,0.9385994210643509)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@4,0.9562458249832999)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@5,0.9666555332887998)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@6,0.974393230906257)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@7,0.9787352482743265)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@8,0.9817969271877087)
16/02/04 02:00:19 INFO TwentyNewsPipeline$: (TRAIN_PRECISION@9,0.9845246047650857)
```

The results may vary slightly in different runs.
