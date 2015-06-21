spark-submit \
--master local[4] \
--executor-memory 4G \
--driver-memory 4G \
--class com.airbnb.aerosolve.demo.ImageImpressionism.JobRunner \
build/libs/image_impressionism-1.0.0-all.jar \
image_impressionism.conf \
$1
