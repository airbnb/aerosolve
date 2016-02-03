spark-submit \
--master local[4] \
--executor-memory 4G \
--driver-memory 4G \
--class com.airbnb.aerosolve.demo.TwentyNews.JobRunner \
build/libs/twenty_news-1.0.0-all.jar \
twenty_news.conf \
$1
