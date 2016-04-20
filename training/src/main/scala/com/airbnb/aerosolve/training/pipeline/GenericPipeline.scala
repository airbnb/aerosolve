package com.airbnb.aerosolve.training.pipeline

import java.io.{BufferedWriter, OutputStreamWriter}

import com.airbnb.aerosolve.core.{Example, FeatureVector, ModelRecord}
import com.airbnb.aerosolve.core.models.{AbstractModel, ForestModel, FullRankLinearModel}
import com.airbnb.aerosolve.core.transforms.Transformer
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.training._
import com.typesafe.config.{Config, ConfigValueFactory}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql._
import org.apache.spark.rdd.RDD
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.apache.spark.SparkContext
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types._

import scala.math.{ceil, max, pow}
import scala.collection.JavaConverters._
import scala.util.{Random, Try}

/*
 * Pipeline for generating, evaluating, and scoring Aerosolve models.
 */
object GenericPipeline {
  val log: Logger = LoggerFactory.getLogger("GenericPipeline")
  val LABEL = "LABEL"

  def makeTrainingRun(sc: SparkContext, config: Config) = {
    val cfg = config.getConfig("make_training")
    val query = cfg.getString("hive_query")
    val output = cfg.getString("output")
    val numShards = cfg.getInt("num_shards")
    val isMulticlass = Try(cfg.getBoolean("is_multiclass")).getOrElse(false)
    val training = makeTraining(sc, query, isMulticlass)

    training
      .coalesce(numShards, true)
      .map(Util.encode)
      .saveAsTextFile(output, classOf[GzipCodec])
  }

  def debugExampleRun(sc: SparkContext, config: Config) = {
    val cfg = config.getConfig("debug_example")
    val query = cfg.getString("hive_query")
    val count = cfg.getInt("count")
    val isMulticlass = Try(cfg.getBoolean("is_multiclass")).getOrElse(false)

    makeTraining(sc, query, isMulticlass)
      .take(count)
      .foreach(logPrettyExample)
  }

  def debugTransformsRun(sc : SparkContext, config : Config) = {
    val cfg = config.getConfig("debug_transforms")
    val query = cfg.getString("hive_query")
    val count = cfg.getInt("count")
    val key = cfg.getString("model_config")
    val isMulticlass = Try(cfg.getBoolean("is_multiclass")).getOrElse(false)

    val input = makeTraining(sc, query, isMulticlass)

    LinearRankerUtils
      .makePointwiseFloat(input, config, key)
      .take(count)
      .foreach(logPrettyExample)
  }

  def trainingRun(
      sc: SparkContext,
      config: Config,
      isTraining: Example => Boolean = isTraining) = {
    val cfg = config.getConfig("train_model")
    val inputPattern = cfg.getString("input")
    val subsample = cfg.getDouble("subsample")
    val modelConfig = cfg.getString("model_config")

    val input = getExamples(sc, inputPattern)
      .filter(isTraining)

    val filteredInput = input.sample(false, subsample)

    TrainingUtils.trainAndSaveToFile(sc, filteredInput, config, modelConfig)
  }

  def getModelAndTransform(
      config : Config,
      modelCfgName : String,
      modelName : String ) = {
    val modelOpt = TrainingUtils.loadScoreModel(modelName)

    if (modelOpt.isEmpty) {
      log.error("Could not load model")
      System.exit(-1)
    }

    val transformer = new Transformer(config, modelCfgName)

    (modelOpt.get, transformer)
  }

  def evalRun(
      sc: SparkContext,
      config : Config, cfgKey : String,
      isTraining: Example => Boolean = isTraining): Unit = {
    val metrics = evalCompute(sc, config, cfgKey, isTraining)

    metrics.foreach(x => log.info(x.toString))
  }

  def evalCompute(
      sc: SparkContext,
      config: Config,
      cfgKey: String,
      isTraining: Example => Boolean): Array[(String, Double)] = {
    val cfg = config.getConfig(cfgKey)
    val modelCfgName = cfg.getString("model_config")
    val modelName = cfg.getString("model_name")
    val inputPattern = cfg.getString("input")
    val subsample = cfg.getDouble("subsample")
    val bins = cfg.getInt("bins")
    val isProb = cfg.getBoolean("is_probability")
    val isRegression = Try(cfg.getBoolean("is_regression")).getOrElse(false)
    val isMulticlass = Try(cfg.getBoolean("is_multiclass")).getOrElse(false)
    val metric = cfg.getString("metric_to_maximize")
    val (model, transformer) = getModelAndTransform(config, modelCfgName, modelName)

    val metrics = evalModelInternal(
      sc,
      transformer,
      model,
      inputPattern,
      subsample,
      bins,
      isProb,
      isRegression,
      isMulticlass,
      metric,
      isTraining
    )

    metrics
  }

  def calibrateRun(
      sc: SparkContext,
      config: Config,
      isTraining: Example => Boolean = isTraining) = {
    val plattsConfig = config.getConfig("calibrate_model")
    val modelCfgName = plattsConfig.getString("model_config")
    val modelName = plattsConfig.getString("model_name")

    val (model, transformer) = getModelAndTransform(config, modelCfgName, modelName)
    val input = plattsConfig.getString("input") // training_data_with_ds
    // get calibration training data
    val data = getExamples(sc, input)
        .sample(false, plattsConfig.getDouble("subsample"))

    val scoresAndLabel = EvalUtil.scoreExamples(sc, transformer, model, data, isTraining, LABEL)

    // Use TRAIN data for train and HOLD data for eval
    val calibrationTraining = scoresAndLabel
      .filter(x => x._2.contains("TRAIN"))
      .map(x => (x._1.toDouble, if(x._2 == "TRAIN_P") true else false))
      .cache()

    val calibrationHoldout = scoresAndLabel
      .filter(x => x._2.contains("HOLD"))
      // Create [score, label] Type: Array[(Double, Boolean)]
      .map(x => (x._1.toDouble, if(x._2 == "HOLD_P") true else false))

    val params = ScoreCalibrator.trainSGD(plattsConfig, calibrationTraining)

    calibrationTraining.unpersist()
    val offset = params(0)
    val slope =  params(1)

    // Evaluation
    val errorTrainCalibrated =
      evalCalibration(calibrationTraining, offset, slope, "")
    val errorTrainNonCalibrated =
      evalCalibration(calibrationTraining, 0, 1, "")
    val errorHoldCalibrated =
      evalCalibration(calibrationHoldout, offset, slope, "")
    val errorHoldNonCalibrated = evalCalibration(calibrationHoldout, 0, 1, "")

    log.info("Number of samples used for training: %d".format(calibrationTraining.count))
    log.info("Training eval result: calibrated %f, non-calibrated %f".format(
      errorTrainCalibrated, errorTrainNonCalibrated)
    )
    log.info("Holdout eval result: calibrated %f, non-calibrated %f".format(
      errorHoldCalibrated, errorHoldNonCalibrated))
    log.info("Calibration parameters: offset = %f slope = %f".format(offset, slope))

    var success = true
    if ((errorTrainNonCalibrated < errorTrainCalibrated) ||
      (errorHoldNonCalibrated < errorHoldCalibrated)) {
      log.error("Calibration is worse than Non-Calibration.")
      success = false
    }

    if (success) {
      // If calibration is successful, update offset and slope of the model
      // otherwise, use the default offset = 0 and slope = 1 in the model
      model.setOffset(offset)
      model.setSlope(slope)
    }

    // Save the model with updated calibration parameters
    try {
      val output: String = plattsConfig.getString("calibrated_model_output")
      val fileSystem = FileSystem.get(new java.net.URI(output), new Configuration())
      val file = fileSystem.create(new Path(output), true)
      val writer = new BufferedWriter(new OutputStreamWriter(file))
      model.save(writer)
      writer.close()
      file.close()
    } catch {
      case _ : Throwable => log.error("Could not save model")
    }
  }

  def modelRecordToString(x: ModelRecord) : String = {
    if (x.weightVector != null && !x.weightVector.isEmpty) {
      "%s\t%s\t%f\t%f\t%s".format(
        x.featureFamily, x.featureName, x.minVal, x.maxVal, x.weightVector.toString)
    } else {
      "%s\t%s\t%f".format(x.featureFamily, x.featureName, x.featureWeight)
    }
  }

  def dumpModelRun(sc: SparkContext, config: Config) = {
    val cfg = config.getConfig("dump_model")
    val modelName = cfg.getString("model_name")
    val modelDump = cfg.getString("model_dump")

    val model = sc
      .textFile(modelName)
      .map(Util.decodeModel)
      .filter(x => x.featureName != null)
      .map(modelRecordToString)

    PipelineUtil.saveAndCommitAsTextFile(model, modelDump)
  }

  def dumpForestRun(sc : SparkContext, config: Config) = {
    val cfg = config.getConfig("dump_forest")
    val modelName = cfg.getString("model_name")
    val modelDump = cfg.getString("model_dump")
    val model = TrainingUtils.loadScoreModel(modelName).get

    val forest = model.asInstanceOf[ForestModel]
    val trees = forest.getTrees().asScala.toArray

    val builder = new StringBuilder()
    val count = trees.size

    for (i <- 0 until count) {
      val tree = trees(i)
      builder ++= tree.toDot().replace("digraph g", "digraph tree_%d".format(i))
    }

    PipelineUtil.writeStringToFile(builder.toString, modelDump)
  }

  def dumpFullRankLinearRun(sc: SparkContext, config: Config) = {
    val cfg = config.getConfig("dump_full_rank_linear_model")
    val modelName = cfg.getString("model_name")
    val modelDump = cfg.getString("model_dump")
    val featuresPerLabel = cfg.getInt("features_per_label")
    val model = TrainingUtils.loadScoreModel(modelName).get.asInstanceOf[FullRankLinearModel]

    val builder = new StringBuilder()

    model.getLabelDictionary.asScala.foreach(entry => {
      val label = entry.getLabel
      val count = entry.getCount
      val index = model.getLabelToIndex.get(label)

      val weights = model.getWeightVector.asScala.flatMap({
        case (family, features) => features.asScala.map({
          case (feature, fv) =>
            Tuple3(family, feature, fv.getValues.apply(index))
        })
      }).toSeq

      // Sort by weight, descending and take top featuresPerLabel
      val sortedWeights = weights.sortBy(entry => -1.0 * entry._3).take(featuresPerLabel)

      sortedWeights.foreach(weightTuple => {
        builder ++= "%s\t%s\t%s\t%f\n".format(
          label, weightTuple._1, weightTuple._2, weightTuple._3
        )
      })
    })

    PipelineUtil.writeStringToFile(builder.toString, modelDump)
  }

  def scoreTableRun(sc: SparkContext, config: Config) = {
    val cfg = config.getConfig("score_table")
    val query = cfg.getString("hive_query")
    val output = cfg.getString("output")
    val numShards = cfg.getInt("num_shards")
    val modelCfgName = cfg.getString("model_config")
    val modelName = cfg.getString("model_name")
    val isMulticlass = Try(cfg.getBoolean("is_multiclass")).getOrElse(false)

    val (model, transformer) = getModelAndTransform(config, modelCfgName, modelName)

    val hc = new HiveContext(sc)
    val hiveTraining = hc.sql(query)

    val schema: Array[StructField] = hiveTraining.schema.fields.toArray
    val lastIdx = schema.size - 1
    if (!schema(lastIdx).name.equals("UNIQUE_ID")) {
      log.error("Last row of the scoring table must be UNIQUE_ID")
      System.exit(-1)
    }

    // What was the schema except for Unique ID
    val origSchema = schema.dropRight(1)

    val modelBC = sc.broadcast(model)
    val transformerBC = sc.broadcast(transformer)

    val examples = hiveTraining
      // ID, example
      .map(x => (x.getString(lastIdx), hiveTrainingToExample(x, origSchema, isMulticlass)))
      .coalesce(numShards, true)

    if (isMulticlass) {
      examples.flatMap(x => {
        scoreMulticlass(x._2, modelBC.value, transformerBC.value)
          .map(result => {
            "%s\t%s\t%f\t%f".format(x._1, result.getLabel, result.getScore, result.getProbability)
          })
      }).saveAsTextFile(output)
    } else {
      examples.map(x => (x._1, score(x._2, modelBC.value, transformerBC.value)))
        .map(x => "%s\t%f\t%f".format(x._1, x._2._1, x._2._2))
        .saveAsTextFile(output)
    }
  }

  def debugScoreTableRun(sc : SparkContext, config: Config) = {
    val cfg = config.getConfig("debug_score_table")
    val query = cfg.getString("hive_query")
    val modelCfgName = cfg.getString("model_config")
    val modelName = cfg.getString("model_name")
    val count = cfg.getInt("count")
    val isMulticlass = Try(cfg.getBoolean("is_multiclass")).getOrElse(false)

    val (model, transformer) = getModelAndTransform(config, modelCfgName, modelName)

    val hc = new HiveContext(sc)
    val hiveTraining = hc.sql(query)
    val schema: Array[StructField] = hiveTraining.schema.fields.toArray
    val lastIdx = schema.size - 1

    if (!schema(lastIdx).name.equals("UNIQUE_ID")) {
      log.error("Last row of the scoring table must be UNIQUE_ID")
      System.exit(-1)
    }

    // What was the schema except for Unique ID
    val origSchema = schema.dropRight(1)

    val ex = hiveTraining
      // ID, example
      .map(x => (x.getString(lastIdx), hiveTrainingToExample(x, origSchema, isMulticlass)))
      .take(count)

    ex.foreach(ex => {
      transformer.combineContextAndItems(ex._2)
      val builder = new java.lang.StringBuilder()
      val score = model.debugScoreItem(ex._2.example.get(0), builder)
      builder.append("Debug score for %s\n".format(ex._1))
      log.info(builder.toString)
    })
  }

  def score(
      example: Example,
      model: AbstractModel,
      transformer: Transformer) = {
    transformer.combineContextAndItems(example)

    val score = model.scoreItem(example.example.get(0))
    val prob = model.scoreProbability(score)

    (score, prob)
  }

  def scoreMulticlass(
      example: Example,
      model: AbstractModel,
      transformer: Transformer) = {
    transformer.combineContextAndItems(example)

    val multiclassResults = model.scoreItemMulticlass(example.example.get(0))
    model.scoreToProbability(multiclassResults)

    multiclassResults.asScala
  }

  private def evalModelInternal(
      sc: SparkContext,
      transformer: Transformer,
      modelOpt: AbstractModel,
      inputPattern: String,
      subSample: Double,
      bins: Int,
      isProb: Boolean,
      isRegression: Boolean,
      isMulticlass: Boolean,
      metric: String,
      isTraining: Example => Boolean) : Array[(String, Double)] = {
    val examples = sc.textFile(inputPattern)
      .map(Util.decodeExample)
      .sample(false, subSample)

    val records = EvalUtil
      .scoreExamplesForEvaluation(
        sc,
        transformer,
        modelOpt,
        examples,
        LABEL,
        isProb,
        isMulticlass,
        isTraining)
      .cache()

    // Find the best F1
    val result = if (isRegression) {
      Evaluation.evaluateRegression(records)
    } else if (isMulticlass) {
      Evaluation.evaluateMulticlassClassification(records)
    } else {
      Evaluation.evaluateBinaryClassification(records, bins, metric)
    }

    records.unpersist()
    result
  }

  def logPrettyExample(ex : Example) = {
    val fv = ex.example.get(0)
    val builder = new StringBuilder()

    builder ++= "\nString Features:"

    if (fv.stringFeatures != null) {
      fv.stringFeatures.asScala.foreach(x => {
        builder ++= "FAMILY : " + x._1 + '\n'
        x._2.asScala.foreach(y => {builder ++= "--> " + y + '\n'})
      })
    }

    builder ++= "\nFloat Features:"

  if (fv.floatFeatures != null) {
      fv.floatFeatures.asScala.foreach(x =>  {
        builder ++= "FAMILY : " + x._1 + '\n'
        x._2.asScala.foreach(y => {builder ++= "--> " + y.toString + '\n'})
      })
    }

    log.info(builder.toString)
  }

  def makeTraining(
      sc: SparkContext,
      query: String,
      isMulticlass: Boolean = false): RDD[Example] = {
    val hc = new HiveContext(sc)
    val hiveTraining = hc.sql(query)
    val schema: Array[StructField] = hiveTraining.schema.fields.toArray

    hiveTraining
      .map(x => hiveTrainingToExample(x, schema, isMulticlass))
  }

  def isTraining(examples : Example) : Boolean = {
    // Take the hash code mod 255 and keep the first 16 as holdout.
    (examples.toString.hashCode & 0xFF) > 16
  }

  def isHoldout(examples : Example) : Boolean = {
    // Take the hash code mod 255 and keep the first 16 as holdout.
    (examples.toString.hashCode & 0xFF) <= 16
  }

  def getExamples(sc : SparkContext, inputPattern : String) : RDD[Example] = {
    val examples : RDD[Example] = sc
      .textFile(inputPattern)
      .map(Util.decodeExample)
    examples
  }

  def evalCalibration(
      input: RDD[(Double, Boolean)],
      offset: Double,
      slope: Double,
      output: String = ""): Double = {
    val scoreLabelProb = input
      // score, label, probability
      .map{x => (x._1, x._2, 1.0 / (1.0 + math.exp(-offset - slope * x._1)))}

    if (output.nonEmpty) {
      // save the (score, label, probability) tuple for offline evaluation
      PipelineUtil
        .saveAndCommitAsTextFile(
          scoreLabelProb.map(x => "%f,%s,%f".format(x._1, x._2, x._3)), output
        )
    }

    val probLabel = scoreLabelProb.map(x => (x._3, x._2)) // RDD[(probability, label)]
    val error = computeCalibrationError(probLabel)

    error
  }

  def computeCalibrationError(input: RDD[(Double, Boolean)]): Double = {
    // input: RDD[(probability, label)], output: calibration error
    val bucketSize = 0.01 // bucket-> (positiveCount, negativeCount)
    val labelsAndPreds = input
        .map{x => {
          val label = x._2
          val probability = x._1
          val bucket = (probability / bucketSize).toLong
          if (label) {
            (bucket, (1, 0))
          } else {
            (bucket, (0, 1))
          }}}
        .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))

    val err = labelsAndPreds
      .map(x => x._2._1.toDouble / (x._2._1 + x._2._2) - x._1 * bucketSize)
      .map(x => x * x)
      .collect()

    Math.sqrt(err.sum / err.length.toDouble)
  }

  def paramSearch(sc : SparkContext, config : Config) = {
    val cfg = config.getConfig("param_search")
    val strategy = cfg.getString("search_strategy")
    val paramCfg = cfg.getConfigList("param_to_tune")
    val paramNames: Array[String] = Try(paramCfg.asScala.map(_.getString("name")).toArray)
      .getOrElse(Array[String]())

    val maxRound: Int = cfg.getInt("max_round")
    val initParamVals: Array[Array[Double]] = Try(
      paramCfg.asScala.map(_.getDoubleList("val").asScala
        .map(_.doubleValue()).toArray).toArray).getOrElse(Array[Array[Double]]())

    if (paramNames.size != initParamVals.size) {
      log.error("incomplete parameter info")
    }

    val paramVals: Array[Array[Double]] = strategy match {
      // TODO (hui_duan) can be extended to adopt more strategies
      case "guided" => initParamVals
      case "grid" => initParamVals.map((x:Array[Double]) => {
        val low = if (x.size > 0) x(0) else 0
        val high = if (x.size > 1) x(1) else 1
        val count = max(ceil(pow(maxRound,1d / initParamVals.size)).toInt, 2)
        val step = (high - low) / (count - 1)
        (0 until count).map(_ * step + low).toArray
      })
      case _ => {
        log.error("Unknown strategy " + strategy)
        Array[Array[Double]]()
      }
    }

    val paramSets : Array[Array[Double]] =
      paramVals.foldLeft(Array[Array[Double]](Array[Double]()))(
        (collector: Array[Array[Double]], ar: Array[Double]) => for (el <- collector; x <- ar)
          yield el :+ x)
    val metrics: Array[(String, Double)] = (if (maxRound >= paramSets.length) paramSets else
      Random.shuffle(paramSets.toList).take(maxRound).toArray).map(
      x => trainEvalForParamSearch(sc, paramNames.zip(x), config))
    val bestMetric = metrics.reduceLeft((x, y) => if (x._2 > y._2) x else y)
    val bestModelOutput = Try(cfg.getString("best_model_output")).getOrElse("")

    if (bestModelOutput.length > 0) {
      PipelineUtil.copyFiles(bestMetric._1, bestModelOutput)
    }

    val recordStarter = "----best trial: " + bestMetric._1 + " : " + bestMetric._2
    val record = metrics.foldLeft(recordStarter)((str: String, x: (String, Double)) =>
      str + "\n----record: " + x._1 + " : " + x._2)

    val outputPath = Try(cfg.getString("output")).getOrElse("")
    if (outputPath.length > 0) {
      PipelineUtil.writeStringToFile(record, outputPath)
    }

    log.info(record)
  }

  def trainEvalForParamSearch(
      sc: SparkContext,
      paramSet: Array[(String,Double)],
      config: Config) : (String, Double) = {
    val cfg = config.getConfig("param_search")
    val modelConfig = cfg.getString("model_config")
    val metricName: String = cfg.getString("metric_to_maximize")
    val paramStr = paramSet.map(x => s"${x._1}_${x._2}").mkString("__")
    val modelPrefix = cfg.getString("model_name_prefix")
    val modelOutput = modelPrefix.concat("__").concat(paramStr).concat(".model")

    // revise train_model.model_config, ${modelConfig}.model_output (for training)
    val baseCfg = config.withValue("train_model.model_config",
      ConfigValueFactory.fromAnyRef(modelConfig))
      .withValue(modelConfig.concat(".").concat("model_output"),
        ConfigValueFactory.fromAnyRef(modelOutput))
      // revise eval_model.model_config, eval_model.metric_to_maximize,
      // eval_model.model_name (for eval)
      .withValue("eval_model.model_config", ConfigValueFactory.fromAnyRef(modelConfig))
      .withValue("eval_model.metric_to_maximize", ConfigValueFactory.fromAnyRef(metricName))
      .withValue("eval_model.model_name", ConfigValueFactory.fromAnyRef(modelOutput))

    // revise ${modelConfig}.${param} for parameter setting
    val finalCfg = paramSet.foldLeft(baseCfg)((base: Config, param: (String, Double)) => {
      base.withValue(
        modelConfig.concat(".").concat(param._1), ConfigValueFactory.fromAnyRef(param._2))
    })

    trainingRun(sc, finalCfg)

    val evalResults = evalCompute(sc, finalCfg, "eval_model", isTraining)

    (modelOutput, evalResults.find(_._1.equals(metricName)).getOrElse(("", -1d))._2)
  }

  def hiveTrainingToExample(
      row: Row,
      schema: Array[StructField],
      isMulticlass: Boolean = false): Example = {
    val example = new Example()
    val featureVector = new FeatureVector()
    example.addToExample(featureVector)

    val stringFeatures = new java.util.HashMap[String, java.util.Set[java.lang.String]]()
    featureVector.setStringFeatures(stringFeatures)

    val floatFeatures = new java.util.HashMap[String, java.util.Map[
      java.lang.String, java.lang.Double]]()
    featureVector.setFloatFeatures(floatFeatures)

    val bias = new java.util.HashSet[java.lang.String]()
    val missing = new java.util.HashSet[java.lang.String]()
    bias.add("B")

    stringFeatures.put("BIAS", bias)
    stringFeatures.put("MISS", missing)

    //val genericFloat = new java.util.HashMap[java.lang.String, java.lang.Double]()
    for (i <- schema.indices) {
      val rowSchema = schema(i)
      val name = rowSchema.name
      val tokens = rowSchema.name.split("_")

      if (tokens.size != 2) {
        if (tokens(0) != LABEL) {
          log.error("Column name not in FAMILY_NAME format or is not LABEL! %s".format(name))
          System.exit(-1)
        }
      }

      val featureFamily = tokens(0)
      val featureName = if (tokens.size > 1) tokens(1) else ""

      if (row.isNullAt(i)) {
        missing.add(name)
      } else {
        rowSchema.dataType match {
          case StringType =>
            val str = row.getString(i)

            if (isMulticlass && featureFamily == LABEL) {
              str.split(",").foreach(classStr => {
                val labelTokens = classStr.split(":").toIndexedSeq

                if (labelTokens.size != 2) {
                  log.error("Multiclass LABEL \"%s\" not in format [label1]:[weight1],...!"
                    .format(str))
                  System.exit(-1)
                }

                val feature = Util.getOrCreateFloatFeature(featureFamily, floatFeatures)

                feature.put(labelTokens(0), labelTokens(1).toDouble)
              })
            } else {
              val feature = Util.getOrCreateStringFeature(featureFamily, stringFeatures)

              if (featureName == "RAW") {
                // In RAW case, don't append feature name
                feature.add(str)
              } else {
                feature.add(featureName + ':' + str)
              }
            }

          case LongType =>
            val lng = row.getLong(i)
            val feature = Util.getOrCreateFloatFeature(featureFamily, floatFeatures)
            feature.put(featureName, lng.toDouble)

          case IntegerType =>
            val int = row.getInt(i)
            val feature = Util.getOrCreateFloatFeature(featureFamily, floatFeatures)
            feature.put(featureName, int.toDouble)

          case FloatType =>
            val dbl = row.getFloat(i)
            val feature = Util.getOrCreateFloatFeature(featureFamily, floatFeatures)
            feature.put(featureName, dbl.toDouble)

          case DoubleType =>
            val dbl = row.getDouble(i)
            val feature = Util.getOrCreateFloatFeature(featureFamily, floatFeatures)
            feature.put(featureName, dbl.toDouble)

          case BooleanType =>
            val bool = row.getBoolean(i)
            val feature = Util.getOrCreateStringFeature(featureFamily, stringFeatures)
            val str = if (bool) "T" else "F"
            feature.add(featureName + ':' + str)
        }
      }
    }

    example
  }
}
