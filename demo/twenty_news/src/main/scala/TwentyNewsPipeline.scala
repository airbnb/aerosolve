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
import scala.collection.JavaConverters._
import scala.collection.JavaConversions

object TwentyNewsPipeline {
  val log: Logger = LoggerFactory.getLogger(this.getClass)

  def debugExampleRun(sc : SparkContext, config : Config) = {
    val inputFile : String = config.getString("input")
    val count : Int = config.getInt("count")
    val example = sc.textFile(inputFile)
      .map(lineToExample)
      .filter(x => x != None)
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
    val strFv = new java.util.HashSet[java.lang.String]()
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

  def isTraining(example : Example) : Boolean = {
    val code : Int = math.abs(example.hashCode()) % 100
    return code < 90
  }

  def trainModel(sc : SparkContext, config : Config) = {
    val trainConfig = config.getConfig("train_model")
    val trainingDataName = trainConfig.getString("input")
    val modelKey = trainConfig.getString("model_key")
    log.info("Training on %s".format(trainingDataName))
    val input = sc.textFile(trainingDataName).map(Util.decodeExample).filter(x => isTraining(x))
    TrainingUtils.trainAndSaveToFile(sc, input, config, modelKey)
  }

  def evalModel(sc : SparkContext, config : Config, key : String) = {
    val testConfig = config.getConfig(key)
    val modelOpt = TrainingUtils.loadScoreModel(testConfig.getString("model_output"))
    if(modelOpt == None) {
      log.error("Could not load the model")
      System.exit(-1)
    }
    val modelKey = testConfig.getString("model_key")
    val transformer = new Transformer(config, modelKey)
    val input : String = testConfig.getString("input")

    val metrics = evalModelInternal(sc, transformer, modelOpt.get, input)
    metrics.foreach(x => log.info(x.toString))
  }

  private def evalModelInternal(sc: SparkContext,
                                transformer: Transformer,
                                modelOpt: AbstractModel,
                                inputPattern: String) : Array[(String, Double)] = {
    val examples = getExamples(sc, inputPattern)

    val records = scoreExamplesForEvaluation(sc, transformer, modelOpt, examples)
    records.take(10).foreach(x => log.info(x.toString))

    Evaluation.evaluateMulticlassClassification(records)
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
                                         examples : RDD[Example]) : RDD[EvaluationRecord] = {
    val modelBC = sc.broadcast(modelOpt)
    val transformerBC = sc.broadcast(transformer)

    examples.map(example => {
      val result = new EvaluationRecord
      result.setIs_training(isTraining(example))
      transformerBC.value.combineContextAndItems(example)
      val score = modelBC.value.scoreItemMulticlass(example.example.get(0)).asScala
      val label = example.example.get(0).floatFeatures.get("LABEL").asScala
      val evalScores = new java.util.HashMap[java.lang.String, java.lang.Double]()
      val evalLabels = new java.util.HashMap[java.lang.String, java.lang.Double]()
      result.setScores(evalScores)
      result.setLabels(evalLabels)
      for (s <- score) {
        evalScores.put(s.label, s.score)
      }
      for (l <- label) {
        evalLabels.put(l._1, l._2)
      }
      result
    })
  }
}
