aerosolve
=========

Machine learning **for humans**.

What is it?
-----------

A machine learning library designed from the ground up to be human friendly.
It is different from other machine learning libraries in the following ways:

  * A [feature transform language](https://github.com/airbnb/aerosolve/tree/master/core/src/main/java/com/airbnb/aerosolve/core/transforms) gives the user a lot of control over the features
  * Human friendly [debuggable models](https://github.com/airbnb/aerosolve/tree/master/core/src/main/java/com/airbnb/aerosolve/core/models)
  * Separate lightweight [Java inference code](https://github.com/airbnb/aerosolve/tree/master/core/src/main/java/com/airbnb/aerosolve/core)
  * Scala code for [training](https://github.com/airbnb/aerosolve/tree/master/training/src/main/scala/com/airbnb/aerosolve/training)
  * Simple [image content analysis code](https://github.com/airbnb/aerosolve/tree/master/core/src/main/java/com/airbnb/aerosolve/core/images) suitable for ordering or ranking images

This library is meant to be used with sparse, interpretable features such as those that commonly occur in search
(search keywords, filters), pricing (listing type, location, price).

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
