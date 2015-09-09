Image Impressionism Demo
========================

# Introduction

In this demo we will train a regression model on the image below in an attempt to recreate it.

![crocus](crocus.jpg)

## Pre-requisites

This demo assumes

  * Spark is installed http://spark.apache.org/downloads.html
  * spark-submit is in your path somewhere
  * Gradle is installed https://gradle.org/
  * Knowing what regression is http://en.wikipedia.org/wiki/Nonparametric_regression
  * Roughly 8 GB of free memory

## Running the demo

The first step is to ensure that the image and output directories are as you want them.
The config file is in src/main/resources/image_impressionism.conf

`emacs src/main/resources/image_impressionism.conf`

Edit the fields image and rootDir to change the input image and output location respectively.
By default the input is the crocus flower image and the output is named output in the current directory.
See https://github.com/typesafehub/config on the syntax of these HOCON files

After you edit the config file, you can build the demo using

`gradle shadowjar --info`

The first step is making the training data from the image. You can do this by running

`sh job_runner.sh MakeTraining`

This should create several part files in output/training_data. The training data tells
the model what intensity it should predict given x,y pixel location and the color channel.

You can view what is in the examples using spark-shell:

```
spark-shell --master local --jars build/libs/image_impressionism-1.0.0-all.jar

scala> import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.core.util.Util

scala> val examples = sc.textFile("output/training_data").map(Util.decodeExample).take(2).foreach(println)
```

The output should look like

```
Example(example:[FeatureVector(stringFeatures:{C=[Red]}, floatFeatures:{$target={=0.5529411764705883}}), FeatureVector(stringFeatures:{C=[Green]}, floatFeatures:{$target={=0.5725490196078431}}), FeatureVector(stringFeatures:{C=[Blue]}, floatFeatures:{$target={=0.5568627450980392}})], context:FeatureVector(stringFeatures:{}, floatFeatures:{LOC={X=0.0, Y=0.0}}))
Example(example:[FeatureVector(stringFeatures:{C=[Red]}, floatFeatures:{$target={=0.5607843137254902}}), FeatureVector(stringFeatures:{C=[Green]}, floatFeatures:{$target={=0.5803921568627451}}), FeatureVector(stringFeatures:{C=[Blue]}, floatFeatures:{$target={=0.5647058823529412}})], context:FeatureVector(stringFeatures:{}, floatFeatures:{LOC={X=0.0, Y=1.0}}))
```

Each training example is composed of two parts - a **context** part which is the same across all the repeated **example**
For instance in the two samples above the location of a pixel is the context and the color channel and target are the items.

Once you are satisfied the training data is OK take a look at the training config in src/main/resources/image_impressionism.conf

You can see the feature engineering for the context, **multiscale grid transform** can be applied once for all three color channels
and then combined using the **cross** transform with each color channel.

The multiscale grid transform is basically like a wavelet decomposition of the image using Haar wavelets.
The cross transform allows us to combine three different color models into one big model and allows
the model to borrow statistical strength from other color channels as well as parent pixel blocks simulatenously.

We can then proceed to train the model using

`sh job_runner.sh TrainModel`

This step should take around 10 minutes and you can watch the progress using [http://localhost:4040/stages/](http://localhost:4040/stages/).

The log should look something like this at the end

```
15/05/18 14:14:42 INFO LinearRankerTrainer: Top 50 weights
15/05/18 14:14:42 INFO LinearRankerTrainer: QLOC : [3.0]=(525.0,858.0) = 0.403138
15/05/18 14:14:42 INFO LinearRankerTrainer: QLOC : [3.0]=(690.0,1005.0) = 0.394892
15/05/18 14:14:42 INFO LinearRankerTrainer: QLOC : [3.0]=(69.0,1302.0) = 0.389529
```

Basically the model has memorized that the brightest spot in the image is around (525, 858)
You can inspect the model using

```
spark-shell --master local --jars build/libs/image_impressionism-1.0.0-all.jar

scala> import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.core.util.Util

scala> val model = sc.textFile("output/model/linear.model").map(Util.decodeModel)

scala> model.filter(x => x.featureName != null && x.featureFamily.contains("_x_")).take(10).foreach(println)
ModelRecord(featureFamily:C_x_QLOC, featureName:Green^[3.0]=(585.0,414.0), featureWeight:0.3104608137334434)
ModelRecord(featureFamily:C_x_QLOC, featureName:Blue^[3.0]=(903.0,1299.0), featureWeight:-0.2862670994597852)
ModelRecord(featureFamily:C_x_QLOC, featureName:Blue^[7.0]=(525.0,1302.0), featureWeight:-0.27945106948968673)
ModelRecord(featureFamily:C_x_QLOC, featureName:Blue^[3.0]=(900.0,1299.0), featureWeight:-0.277188942233763)
```

This tells us that the greenest pixel is at (585, 414) and the least blue pixel is at (903, 1299)

We are now ready to score the model! Type in

```
sh job_runner.sh MakeImpression
open output/impression.jpg
```

You should then see the image below which is an approximation of the image with the smallest block being 3x3 block.
We also extrapolate the lower part of the image a bit so you can see what extrapolation does.

![impression](impression.jpg)

Here is a movie of how the model adds stumps from 3% to 100% capacity

![movie](movie.gif)
