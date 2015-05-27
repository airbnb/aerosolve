package com.airbnb.aerosolve.demo.IncomePrediction;
import com.typesafe.config.ConfigFactory
import org.apache.spark.{SparkContext, SparkConf}
import org.slf4j.{LoggerFactory, Logger}

/*
 * Runs an arbitrary job given a config resource name.
 * The config must contain
 * job_name : name_of_job
 * jobs = [ list of jobs ]
 * ... other job specific configs.
 * Example command line:
 * bin/spark-submit --executor-memory 8G
 * --class com.airbnb.aerosolve.demo.ImageImpressionism.JobRunner
 * image_impressionism-0.1.2-all.jar
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
          case "MakeTraining" => IncomePredictionPipeline
            .makeExampleRun(sc, config.getConfig("make_training"))
          case "MakeTesting" => IncomePredictionPipeline
            .makeExampleRun(sc, config.getConfig("make_testing"))
          case "TrainModel" => IncomePredictionPipeline
            .trainModel(sc, config)
          case "EvalTesting" => IncomePredictionPipeline
            .evalModel(sc, config, "eval_testing")
          case "EvalTraining" => IncomePredictionPipeline
            .evalModel(sc, config, "eval_training")
          case _ => log.error("Unknown job " + job)
        }
      } catch {
        case e : Exception => log.error("Exception on job %s : %s".format(job, e.toString))
          System.exit(-1)
      }
    }
  }
}
