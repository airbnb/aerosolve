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

## Running the demo

The first step is to ensure that the image and output directories are as you want them.
The config file is in src/main/resources/image_impressionism.conf

`emacs src/main/resources/image_impressionism.conf`

Edit the fields image and rootDir to change the input image and output location respectively.
By default the input is the crocus flower image and the output is named output in the current directory.
See https://github.com/typesafehub/config on the syntax of these HOCON files

After you edit the config file, you can build the demo using

`gradle shadowjar --info`
