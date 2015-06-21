spark-submit \
--master local[4] \
--executor-memory 4G \
--driver-memory 4G \
--class com.airbnb.aerosolve.demo.IncomePrediction.JobRunner \
build/libs/income_prediction-1.0.0-all.jar \
income_prediction.conf \
$1
