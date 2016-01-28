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

The results may vary slightly in different runs.
