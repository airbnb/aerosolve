aerosolve
=========

Machine learning **for humans**.

What is it?
-----------

A machine learning library designed from the ground up to be human friendly.
It is different from other machine learning libraries in the following ways:

  * A [thrift based feature representation](https://github.com/airbnb/aerosolve/tree/master/core/src/main/thrift) that enables pairwise ranking loss and single context multiple item representation.
  * A [feature transform language](https://github.com/airbnb/aerosolve/tree/master/core/src/main/java/com/airbnb/aerosolve/core/transforms) gives the user a lot of control over the features
  * Human friendly [debuggable models](https://github.com/airbnb/aerosolve/tree/master/core/src/main/java/com/airbnb/aerosolve/core/models)
  * Separate lightweight [Java inference code](https://github.com/airbnb/aerosolve/tree/master/core/src/main/java/com/airbnb/aerosolve/core)
  * Scala code for [training](https://github.com/airbnb/aerosolve/tree/master/training/src/main/scala/com/airbnb/aerosolve/training)
  * Simple [image content analysis code](https://github.com/airbnb/aerosolve/tree/master/core/src/main/java/com/airbnb/aerosolve/core/images) suitable for ordering or ranking images

This library is meant to be used with sparse, interpretable features such as those that commonly occur in search
(search keywords, filters), pricing (listing type, location, price). It is not as interpretable with problems with very dense
non-human interpretable features such as raw pixels or audio samples.

The are a few reasons to focus on interpretability:

  * Your corpus is new and not fully defined and you want more insight into your corpus
  * Having interpretable models lets you iterate quickly. Figure out where the model disagrees most and have insight into what kind of new features are needed.
  * Debugging noisy features. By plotting the feature weights you can discover buggy features and 
  * You can discover relationships between different variables and your target prediction. e.g. Plotting [graphs of reviews and 3-star reviews](airbnb.github.io/aerosolve/) is more interpretable than many nested if then else rules.

How to get started?
-------------------

Check out the image impression demo where you can learn how to teach
the algorithm to paint in the pointilism style of painting
[Image Impressionism Demo](https://github.com/airbnb/aerosolve/tree/master/demo/image_impressionism)

There is also an income prediction demo based on a popular
machine learning benchmark
[Income Prediction Demo](https://github.com/airbnb/aerosolve/tree/master/demo/income_prediction)

Support
-------

User group : https://groups.google.com/forum/#!forum/aerosolve-users
