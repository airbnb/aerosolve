#!/usr/bin/env bash
# rename training-all.jar to aerosolve-generic-pipeline-current.jar
# set driver-memory and executor-memory for your job.
# ./demo.bash demo_train.conf GenericDebugExamples
# if you have additional java options (options pass to conf) i.e. ds=2016-01-01
# ./demo.bash demo_train.conf GenericDebugExamples ds=2016-01-01
# echo $SPARK_HOME to see if spark is defined, if not, ask admin for spark location.
# i.e. inside airbnb cluster, use /usr/bin/abb-spark-15-submit

$SPARK_HOME/spark-submit \
--master yarn-client \
--conf spark.akka.frameSize=800 \
--conf spark.akka.timeout=300 \
--conf spark.cleaner.ttl=36000 \
--conf spark.core.connection.ack.wait.timeout=720 \
--conf spark.default.parallelism=1000 \
--conf spark.task.maxFailures=7 \
--conf spark.yarn.executor.memoryOverhead=2048 \
--driver-class-path . \
--driver-java-options "$3" \
--driver-memory 16G \
--executor-cores 4 \
--executor-memory 16G \
--num-executors 250 \
--class com.airbnb.aerosolve.training.pipeline.JobRunner \
aerosolve-generic-pipeline-current.jar  \
"$1" "$2"
