package com.airbnb.aerosolve.demo.IncomePrediction;

import com.airbnb.aerosolve.core.{EvaluationRecord, Example, ModelRecord, FeatureVector}
import com.airbnb.aerosolve.core.models.AbstractModel
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.core.transforms.Transformer
import com.airbnb.aerosolve.training.{Evaluation, TrainingUtils}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.slf4j.{LoggerFactory, Logger}
import com.typesafe.config.Config
import org.apache.hadoop.io.compress.GzipCodec

object IncomePredictionPipeline {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  def makeExampleRun(sc : SparkContext, config : Config) = {
    val inputFile : String = config.getString("input")
    val outputFile : String = config.getString("output")
    log.info("Reading training data %s".format(inputFile))
    log.info("Writing training data to %s".format(outputFile))
    val source = scala.io.Source.fromFile(inputFile)
    val lines = try source.mkString.split('\n') finally source.close()
    val example = sc.parallelize(lines)
      // Array[String]: age, workclass, fnlwgt, eduction, eduction-num, marital-status, occupation
      // relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country
      .map(line => line.split(",", -1).map(_.trim))
      .filter(line => line.length == 15)
      // RDD[Example]
      .map(lineToExample)
      .map(Util.encode)
      .saveAsTextFile(outputFile, classOf[GzipCodec])
  }

  def lineToExample(line: Array[String]): Example = {
    val example = new Example()
    val featureVector = new FeatureVector()
    example.addToExample(featureVector)
    // set string features
    val stringFeatures = new java.util.HashMap[String, java.util.Set[java.lang.String]]()
    featureVector.setStringFeatures(stringFeatures)
    val strFv = new java.util.HashSet[java.lang.String]() // education features
    val bias = new java.util.HashSet[java.lang.String]()
    bias.add("B")
    stringFeatures.put("S", strFv)
    stringFeatures.put("BIAS", bias)
    // set float features
    val floatFeatures = new java.util.HashMap[String, java.util.Map[java.lang.String, java.lang.Double]]()
    featureVector.setFloatFeatures(floatFeatures)
    val floatFv = new java.util.HashMap[java.lang.String, java.lang.Double]()
    val label = new java.util.HashMap[java.lang.String, java.lang.Double]()
    floatFeatures.put("F", floatFv)
    floatFeatures.put("$rank", label)
    // line(0): age
    floatFv.put("age", line(0).toDouble)
    // line(1): workclass
    strFv.add(line(1))
    // line(2): fnlwgt
    floatFv.put("fnlwgt", line(2).toDouble)
    // line(3): education
    strFv.add(line(3))
    // line(4): eduction-num
    floatFv.put("edu-num", line(4).toDouble)
    // line(5): marital-status
    strFv.add(line(5))
    // line(6): occupation
    strFv.add(line(6))
    // line(7): relationship
    strFv.add(line(7))
    // line(8): race
    strFv.add(line(8))
    // line(9): sex
    strFv.add(line(9))
    // line(10): capital-gain
    floatFv.put("capital-gain", line(10).toDouble)
    // line(11): capital-loss
    floatFv.put("capital-loss", line(11).toDouble)
    // line(12): hours-per-week
    floatFv.put("hours", line(12).toDouble)
    // line(13): native-country
    strFv.add(line(13))
    // line(14): label=1 if ">50K" else -1
    label.put("", if(line(14).contains(">")) 1.0 else -1.0)

    example
  }

  def trainModel(sc : SparkContext, config : Config) = {
    val trainConfig = config.getConfig("train_model")
    val trainingDataName = trainConfig.getString("input")
    val modelKey = trainConfig.getString("modelKey")
    log.info("Training on %s".format(trainingDataName))
    val input = sc.textFile(trainingDataName).map(Util.decodeExample)
    TrainingUtils.trainAndSaveToFile(sc, input, config, modelKey)
  }

  def evalModel(sc : SparkContext, config : Config, key : String) = {
    val testConfig = config.getConfig(key)
    val modelOpt = TrainingUtils.loadScoreModel(testConfig.getString("model_output"))
    if(modelOpt.isEmpty) {
      log.error("Could not load the model")
      System.exit(-1)
    }
    val modelKey = testConfig.getString("modelKey")
    val transformer = new Transformer(config, modelKey)
    val input : String = testConfig.getString("input")
    val subsample : Double = testConfig.getDouble("subsample")
    val bins : Int = testConfig.getInt("bins")
    val isTraining : Boolean = testConfig.getBoolean("is_training")
    val resultsOutputPath : String = testConfig.getString("results_output_path")

    val metrics = evalModelInternal(sc, transformer, modelOpt.get, input, Some(subsample), bins, isTraining, resultsOutputPath)
    metrics.foreach(x => log.info(x.toString))
  }

  private def evalModelInternal(sc: SparkContext,
                                transformer: Transformer,
                                modelOpt: AbstractModel,
                                inputPattern: String,
                                subSample: Option[Double],
                                bins : Int,
                                isTraining : Boolean,
                                resultsOutputPath : String) : Array[(String, Double)] = {
    val examples =
      if (subSample.isDefined) {
        getExamples(sc, inputPattern).sample(false, subSample.get)
      } else {
        getExamples(sc, inputPattern)
      }

    val records = scoreExamplesForEvaluation(sc, transformer, modelOpt, examples, isTraining).cache()

    // Find the best F1
    Evaluation.evaluateBinaryClassificationWithResults(records, bins, "!HOLD_F1", resultsOutputPath)
  }

  private def getExamples(sc : SparkContext, inputPattern : String) : RDD[Example] = {
    val examples : RDD[Example] = sc
      .textFile(inputPattern)
      .map(Util.decodeExample)
    examples
  }

  private def scoreExamplesForEvaluation(sc: SparkContext,
                                         transformer: Transformer,
                                         modelOpt: AbstractModel,
                                         examples : RDD[Example],
                                         isTraining : Boolean) : RDD[EvaluationRecord] = {
    val modelBC = sc.broadcast(modelOpt)
    val transformerBC = sc.broadcast(transformer)

    examples.map(example => {
      val result = new EvaluationRecord
      result.setIs_training(isTraining)
      transformerBC.value.combineContextAndItems(example)
      val score = modelBC.value.scoreItem(example.example.get(0))
      val prob = modelBC.value.scoreProbability(score)
      val rank = example.example.get(0).floatFeatures.get("$rank").get("")
      result.setScore(prob)
      result.setLabel(rank)
      result
    })
  }
}
