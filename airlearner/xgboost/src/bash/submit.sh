#!/usr/bin/env bash

# sample usage: ./submit.sh task_name config_name > log_name 2>&1
if [ $# -ne 2 ]; then
    echo "Usage: $0 [task] [config file] [Optional options]"
    exit
fi

# replace class and jar with your own jar name and class name.
# spark.default.parallelism = round * min_round (xgboost/search.conf)
/usr/bin/abb-spark-16-submit \
--master yarn-client \
--conf spark.default.parallelism=6496 \
--conf spark.shuffle.service.enabled=true \
--conf spark.akka.frameSize=800 \
--conf spark.task.maxFailures=700 \
--conf spark.cleaner.ttl=172800 \
--conf spark.core.connection.ack.wait.timeout=6000 \
--conf spark.driver.maxResultSize=4G \
--conf spark.yarn.executor.memoryOverhead=22528 \
--conf spark.yarn.driver.memoryOverhead=4096 \
--conf spark.shuffle.memoryFraction=0.2 \
--conf spark.storage.memoryFraction=0.3 \
--conf spark.sql.shuffle.partitions=100 \
--conf spark.speculation=false \
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
--conf spark.kryoserializer.buffer.max=256m \
--conf spark.shuffle.manager=tungsten-sort \
--conf spark.shuffle.service.enabled=true \
--conf spark.network.timeout=1200 \
--num-executors 100 \
--executor-cores 1 \
--executor-memory 2G \
--driver-memory 8G \
--class com.airbnb.airbnb.Runner \
job.jar \
"$1" "$2"
