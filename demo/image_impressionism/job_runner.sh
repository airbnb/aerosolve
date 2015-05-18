spark-submit \
--master local[4] \
--executor-memory 8G \
--class com.airbnb.aerosolve.demo.ImageImpressionism.JobRunner \
build/libs/image_impressionism-0.1.2-all.jar \
image_impressionism.conf \
$1