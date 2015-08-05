package com.airbnb.aerosolve.demo.ImageImpressionism;

import org.apache.spark.{SparkContext, SparkConf}
import org.slf4j.{LoggerFactory, Logger}
import com.typesafe.config.ConfigFactory

/*
 * Runs an arbitrary job given a config resource name.
 * The config must contain
 * job_name : name_of_job
 * jobs = [ list of jobs ]
 * ... other job specific configs.
 * Example command line:
 * bin/spark-submit --executor-memory 8G
 * --class com.airbnb.aerosolve.demo.ImageImpressionism.JobRunner
 * image_impressionism-1.0.0-all.jar
 * image_impressionism.conf
 */

object JobRunner {
  def main(args: Array[String]): Unit = {
    val log: Logger = LoggerFactory.getLogger("Job.Runner")
    if (args.length < 2) {
      log.error("Usage: Job.Runner config_name job1,job2...")
      System.exit(-1)
    }
    log.info("Loading config from " + args(0))
    val config = ConfigFactory.load(args(0))
    val jobs : Seq[String] = args(1).split(',')
    val conf = new SparkConf().setAppName("ImageImpressionism")
    val sc = new SparkContext(conf)
    for (job <- jobs) {
      log.info("Running " + job)
      try {
        job match {
          case "MakeTraining" => ImageImpressionismPipeline
            .makeTrainingRun(sc, config.getConfig("make_training"))
          case "TrainModel" => ImageImpressionismPipeline
            .trainModel(sc, config)
          case "MakeImpression" => ImageImpressionismPipeline
            .makeImpression(sc, config)
          case "MakeMovie" => ImageImpressionismPipeline
            .makeMovie(sc, config)
          case _ => log.error("Unknown job " + job)
        }
      } catch {
        case e : Exception => log.error("Exception on job %s : %s".format(job, e.toString))
        System.exit(-1)
      }
    }
  }
}
