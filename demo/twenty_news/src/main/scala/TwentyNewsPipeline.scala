package com.airbnb.aerosolve.demo.TwentyNews;

import com.airbnb.aerosolve.core.{EvaluationRecord, Example, ModelRecord, FeatureVector}
import com.airbnb.aerosolve.core.models.AbstractModel
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.core.transforms.Transformer
import com.airbnb.aerosolve.training.{Evaluation, TrainingUtils, LinearRankerUtils}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.slf4j.{LoggerFactory, Logger}
import com.typesafe.config.Config
import org.apache.hadoop.io.compress.GzipCodec

object TwentyNewsPipeline {
  val log: Logger = LoggerFactory.getLogger(this.getClass)

  def debugExampleRun(sc : SparkContext, config : Config) = {
    val inputFile : String = config.getString("input")
    val count : Int = config.getInt("count")
    val example = sc.textFile(inputFile)
      .map(lineToExample)
      .filter(x => x !=	None)
      .map(x => x.get)
      .take(count)
      .foreach(x => log.info(x.toString))
  }

  def makeExampleRun(sc : SparkContext, config : Config) = {
    val inputFile : String = config.getString("input")
    val outputFile : String = config.getString("output")
    log.info("Reading training data %s".format(inputFile))
    log.info("Writing training data to %s".format(outputFile))
    val example = sc.textFile(inputFile)
      .map(lineToExample)
      .filter(x => x != None)
      .map(x => x.get)
      .map(Util.encode)
      .saveAsTextFile(outputFile, classOf[GzipCodec])
  }

  def debugTransformRun(sc : SparkContext, baseConfig : Config, config : Config) = {
    val inputFile : String = config.getString("input")
    val count : Int = config.getInt("count")
    val key = config.getString("model_key")
    val example = sc.textFile(inputFile)
      .map(lineToExample)
      .filter(x => x != None)
      .map(x => x.get)

    LinearRankerUtils
      .makePointwiseFloat(example, baseConfig, key)
      .take(count)
      .foreach(x => log.info(x.toString))
  }

  def lineToExample(line: String) : Option[Example] = {
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
    val labelFv = new java.util.HashMap[java.lang.String, java.lang.Double]()
    floatFeatures.put("LABEL", labelFv)

    val tokens = line.split("\t")
    if (tokens.size != 2) return None

    // The label
    labelFv.put(tokens(0), 1.0)

    // The document
    strFv.add(tokens(1))

    Some(example)
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
    if(modelOpt == None) {
      log.error("Could not load the model")
      System.exit(-1)
    }
    val modelKey = testConfig.getString("modelKey")
    val transformer = new Transformer(config, modelKey)
    val input : String = testConfig.getString("input")
    val subsample : Double = testConfig.getDouble("subsample")
    val bins : Int = testConfig.getInt("bins")
    val isTraining : Boolean = testConfig.getBoolean("is_training")

    val metrics = evalModelInternal(sc, transformer, modelOpt.get, input, Some(subsample), bins, isTraining)
    metrics.foreach(x => log.info(x.toString))
  }

  private def evalModelInternal(sc: SparkContext,
                                transformer: Transformer,
                                modelOpt: AbstractModel,
                                inputPattern: String,
                                subSample: Option[Double],
                                bins : Int,
                                isTraining : Boolean) : Array[(String, Double)] = {
    val examples =
      if (subSample != None) {
        getExamples(sc, inputPattern).sample(false, subSample.get)
      } else {
        getExamples(sc, inputPattern)
      }

    val records = scoreExamplesForEvaluation(sc, transformer, modelOpt, examples, isTraining).cache()

    // Find the best F1
    Evaluation.evaluateBinaryClassification(records, bins, "!HOLD_F1")
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
