package com.airbnb.aerosolve.training.strategy.trainer

import com.airbnb.aerosolve.training.pipeline.PipelineUtil
import com.airbnb.aerosolve.training.strategy.config.{BaseSearchConfig, DirectQueryEvalConfig, StrategyModelEvalConfig, TrainingOptions}
import com.airbnb.aerosolve.training.strategy.data.{BinaryTrainingSample, ModelOutput, TrainingData}
import com.airbnb.aerosolve.training.strategy.eval.BinaryMetrics
import com.airbnb.aerosolve.training.strategy.params.StrategyParams
import com.airbnb.aerosolve.training.utils.{DateTimeUtil, HiveUtil, Sort}
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel
import org.slf4j.{Logger, LoggerFactory}

import scala.reflect.ClassTag
import scala.util.{Random, Try}

trait BinaryTrainer [T <: BinaryTrainingSample] extends Serializable {
  val log: Logger = LoggerFactory.getLogger(this.getClass.getName)

  def strategyParams: StrategyParams[T]
  def trainingData: TrainingData[T]

  def getDefaultParams(trainingOptions: TrainingOptions): StrategyParams[T] = {
    if (strategyParams.params.length == 0) {
      strategyParams.getDefaultParams(trainingOptions)
    } else {
      strategyParams
    }
  }

  def loadParamsFromHive(hiveContext: HiveContext, query: String): RDD[(String, StrategyParams[T])] = {
    hiveContext.sql(query)
      // TODO, generalize for all array params
      .map(row => (parseKeyFromHiveRow(row),
      strategyParams.parseParamsFromHiveRow(row)))
  }

  def filter(length: Int): Boolean = {
    length > BinaryTrainer.MIN_COUNT
  }

  def getLearningRate(r0: Double, r1: Double,
                      example: T,
                      options: TrainingOptions): Double

  // each trainer provides its own schema to generate data frame
  def createDataFrameFromModelOutput(models: RDD[(String, StrategyParams[T])], hc: HiveContext): DataFrame

  // Training data load from either hive or data frame object.
  def loadDataFromHive(hiveContext: HiveContext,
                       dataQuery: String)(implicit c: ClassTag[T]):
      RDD[(String, Seq[T])] = {
    loadDataFromDataFrame(hiveContext.sql(dataQuery))
  }

  // TODO refactor key to Long instead of String
  // That would improve join performance a little for both space efficiency
  // and comparator runtime efficiency.
  def parseKeyFromHiveRow(row: Row): String

  def loadModelWithIndexMapFromHdfs(output: String):
  scala.collection.Map[(java.lang.String, Int), StrategyParams[T]] = {
    val modelStr = PipelineUtil.readStringFromFile(output)
    modelStr
      .split("\n")
      .map(strategyParams.parseLine)
      .toMap
  }

  // load Training data by join two DataFrame with key,
  // this generates new training samples which
  // should has same fields as the loadTrainingDataFromHive methods
  // use this method if you want to change any fields and retrain model
  // TODO wrap keys inside trainer, i.e. Tanh Trainer use g_id_listing
  def loadTrainingDataByJoinTwoDataFrames(main: DataFrame,
                                          key: Seq[String],
                                          addition: DataFrame)(implicit c: ClassTag[T]):
  RDD[(String, Seq[T])] = {
    val joined = main.join(addition, key)
    loadDataFromDataFrame(joined)
  }

  def loadDataFromDataFrame(data: DataFrame)(implicit c: ClassTag[T]):
      RDD[(String, Seq[T])] = {
    HiveUtil.loadDataFromDataFrameGroupByKey(
      data,
      parseKeyFromHiveRow,
      trainingData.parseSampleFromHiveRow)
  }

  // default is same as loadTrainingDataFromHive
  // allow to override so that it loads different data
  // refer to StrategyModelTrainerV1.loadDataWithOptions for how to use it
  // TODO retire it
  def loadDataWithOptions(config: Config,
                          hc: HiveContext,
                          dataQuery: String,
                          forTraining: Boolean = true)(implicit c: ClassTag[T]): RDD[(String, Seq[T])] = {
    loadDataFromHive(hc, dataQuery)
  }

  def getEmptyResultRDD(sc: SparkContext): RDD[(String, StrategyParams[T])] = {
    sc.emptyRDD
  }

  // return trained result in data frame,
  // i.e. StrategyModelTrainerV1 return StrategyModelDataSource.schema
  def getResultDataFrame(strategyConfig: Config,
                         hc: HiveContext,
                         trainingData: RDD[(String, Seq[T])])(implicit c: ClassTag[T]): DataFrame = {
    val models: RDD[(String, StrategyParams[T])] =
      trainStrategyModelWithRDD(
        hc,
        strategyConfig,
        trainingData
      )
    createDataFrameFromModelOutput(models, hc)
  }

  def trainStrategyModelWithRDD(hc: HiveContext,
                                config: Config,
                                rawTrainingData: RDD[(String, Seq[T])]
                               )(implicit c: ClassTag[T]): RDD[(String, StrategyParams[T])]= {
    val trainingOptions = TrainingOptions.loadBaseTrainingOptions(
      config.getConfig("training_options"))
    // prepare training data
    val trainingExamples = joinTrainingSamplesWithParams(hc, config, rawTrainingData, trainingOptions)
    trainAll(trainingExamples, trainingOptions)
  }

  // support parameter search, i.e. given Array[TrainingOptions]
  // return trained models for each TrainingOption in the array
  // key in the returning RDD is (id_listing, param_idx),
  // where param_idx specifies the index of the TrainingOptions used to run the training
  def trainAllWithOptionArray(examples: RDD[(String, Seq[T])],
                              optionArr: Array[TrainingOptions]
                             ): RDD[((String, Int), StrategyParams[T])] = {
    // Note: This does not support initialization from existing parameters
    examples
      .mapPartitions(iterable => {
        iterable.flatMap { example =>
          val id: String = example._1
          val samples: Seq[T] = example._2
          optionArr.zipWithIndex.map { case (option, idx) =>
            ((id, idx), train(samples, option))
          }.iterator
        }
      }
      )
  }

  def trainAll(examples: RDD[(String, (Seq[T], Option[StrategyParams[T]]))],
               options: TrainingOptions): RDD[(String, StrategyParams[T])] = {
    examples
      .map { case (id: String,
      (samples: Seq[T], param: Option[StrategyParams[T]])) =>
        (id, train(samples, options, param))
      }
  }

  def trainWithParamSearchPerModel(
      sc: SparkContext,
      config: Config)(implicit c: ClassTag[T]): Unit = {
    val searchConf = BaseSearchConfig.loadConfig(config)
    // prepare training data
    val hc = new HiveContext(sc)
    val evalConfig = DirectQueryEvalConfig.loadConfig(hc, config, trainingData)

    trainWithBaseSearchConfigPerModel(sc, searchConf, evalConfig)
  }

  def trainWithBaseSearchConfigPerModel(
      sc: SparkContext,
      searchConfig: BaseSearchConfig,
      evalConfig: StrategyModelEvalConfig)(implicit c: ClassTag[T]): Unit = {
    // get strategy model training data
    val hc = new HiveContext(sc)

    val evalSamples = loadDataFromHive(
      hc, evalConfig.evalDataQuery)
      .persist(StorageLevel.MEMORY_AND_DISK)

    val trainingOptionsArray: Array[TrainingOptions] = searchConfig.getTrainingOptions

    log.info(s"Load Training Data ${trainingOptionsArray.mkString(",")}")
    val trainingExamples: RDD[(String, Seq[T])] = loadDataFromHive(
      hc,
      evalConfig.trainingDataQuery)
      .filter(x => filter(x._2.length))
      // Shuffle distributes data evenly and with each partition has a lot less data
      // executor is less likely to spill. If you have 500+ executor cores, it is always
      // good to shuffle. Otherwise the cost of shuffling might be big enough and the lack
      // of parallelism makes it not worth it.
      //
      // NB: If uniform sampling is performed via simple mod, caution should be exercised here
      // to make sure the hashcode can be uniformed distributed.
      // For example, the current implementation is a String based hashcode which would not correlate
      // with simple mod on id_listing.
      // However, if we switch to Long based key, then the repartition number should be co-prime
      // with modular used in sampling.
      .coalesce(evalConfig.partitions, evalConfig.shuffle)
      .persist(StorageLevel.MEMORY_AND_DISK)

    log.info("Training models")
    val result = searchBestOptionsPerModel(trainingExamples, evalSamples, trainingOptionsArray)
    ModelOutput.save(hc, result, searchConfig.table, searchConfig.partition)
  }

  def searchBestOptionsPerModel(trainingExamples: RDD[(String, Seq[T])],
                                evalExamples: RDD[(String, Seq[T])],
                                optionArr: Array[TrainingOptions]
                          ): RDD[ModelOutput[T]] = {
    // Note: This does not support initialization from existing parameters
    trainingExamples.join(evalExamples)
      .map( example => {
          val id: String = example._1
          val training: Seq[T] = example._2._1
          val eval = example._2._2
          searchBestOptions(id, training, eval, optionArr)
        }
      )
  }

  def searchBestOptions(id: String,
                        trainingExamples: Seq[T],
                        evalExamples: Seq[T],
                        optionArr: Array[TrainingOptions]): ModelOutput[T] = {
     val result = optionArr.map { option =>
      (trainAndEval(trainingExamples, evalExamples, option), option)
     }.min(Ordering.by{
       output:((StrategyParams[T], Double), TrainingOptions) =>
         output._1._2
     })
     val params = result._1._1
     val metrics = BinaryTrainer.getMetrics(evalExamples, params)
     val loss = result._1._2
     val option = result._2
     ModelOutput(id, params, loss, metrics, option)
  }

  def trainAndEval(trainingExamples: Seq[T],
                   evalExamples: Seq[T],
                   defaultOptions: TrainingOptions,
                   inputParam: Option[StrategyParams[T]] = None,
                   debug: Boolean = false): (StrategyParams[T], Double) = {
    val params = train(trainingExamples, defaultOptions, inputParam, debug)
    (params, eval(evalExamples, params, defaultOptions.evalRatio))
  }

  def eval(examples: Seq[T],
           params: StrategyParams[T],
           ratio: Double): Double = {
    val evalResult = examples.map( example =>
      BinaryTrainer.evalExample(example, params, ratio)
    )
    evalResult.sum / evalResult.length
  }

  // TODO research training converge @yitong
  def train(examples: Seq[T],
            defaultOptions: TrainingOptions,
            inputParam: Option[StrategyParams[T]] = None,
            debug: Boolean = false): StrategyParams[T] = {
    // Input:
    // - examples are corresponding to the same listing
    // - each implementation defines its own params
    // Output:
    // - updated host rejection parameters
    var params = getOrDefaultParams(defaultOptions, inputParam)
    if (BinaryTrainer.enoughTrueSamples(defaultOptions, examples)) {
      var r0 = defaultOptions.r0
      var r1 = defaultOptions.r1
      var minLossSum = Double.MaxValue
      var bestParam = params
      val length = examples.length
      val batchNum = Math.floor(length / defaultOptions.miniBatchSize).toInt
      for (i <- 0 until defaultOptions.numEpochs) {
        // sample examples for training
        val samplesShuffled: Seq[T] = Random.shuffle(examples)
        var lossSum = 0.0
        for (j <- 0 until batchNum) {
          val samples = samplesShuffled.slice(j * defaultOptions.miniBatchSize, (j + 1) * defaultOptions.miniBatchSize)
          val (loss, newParams) = trainMiniBatch(params, samples, defaultOptions, r0, r1)
          lossSum += loss
          params = newParams
        }

        if (lossSum < minLossSum) {
          minLossSum = lossSum
          bestParam = params
        }

        if (debug) {
          val avgLossSum = lossSum / length
          log.info(s"Epoch $i, lossSum=$lossSum, avgLossSum=$avgLossSum" + bestParam.prettyPrint)
        }

        r0 = r0 * defaultOptions.rateDecay
        r1 = r1 * defaultOptions.rateDecay
      }

      val minAvgLossSum = minLossSum / length

      if (acceptLoss(minAvgLossSum, defaultOptions) && bestParam.hasValidValue) {
        bestParam
      } else {
        params
      }
    } else {
      params
    }
  }

  private def acceptLoss(loss: Double, option:TrainingOptions): Boolean = {
    if (option.maxAvgLossRatio > 0) {
      loss <= option.maxAvgLossRatio
    } else {
      loss <= BinaryTrainer.maxAvgLossSum
    }
  }

  // join previous run's model params if initialize_from_global_prior conf is set
  def joinTrainingSamplesWithParams(hc: HiveContext,
                                    config: Config,
                                    rawTrainingData: RDD[(String, Seq[T])],
                                    trainingOptions: TrainingOptions
                                   )(implicit c: ClassTag[T]): RDD[(String, (Seq[T], Option[StrategyParams[T]]))] = {
    // prepare training data
    val trainingSamples = rawTrainingData
      .filter(x => filter(x._2.length))

    // if initializeFromGlobalPrior is true, we initialize the model using global prior
    // otherwise, we initialize the model using previous day's output
    val initializeFromGlobalPrior = Try(config.getBoolean("initialize_from_global_prior"))
      .getOrElse(true)

    if (initializeFromGlobalPrior) {
      val emptyStrategyParamInput: Option[StrategyParams[T]] = None
      trainingSamples
        .mapValues(sample => (sample, emptyStrategyParamInput))
    } else {
      // initialize using previous day's output
      val dsEval = config.getString("ds_eval")
      val paramsQuery = config.getString("params_query")
      val params = loadParamsFromHive(hc,
        paramsQuery.replace("$DS_EVAL", DateTimeUtil.dateMinus(dsEval, 1)))

      trainingSamples.leftOuterJoin(params)
    }
  }

  def trainMiniBatch(params: StrategyParams[T],
                     samples: Seq[T],
                     options: TrainingOptions,
                     r0: Double,
                     r1: Double): (Double, StrategyParams[T]) = {
    var lossSum = 0.0
    val gradSum = Array.fill[Double](params.params.length)(0)
    samples
      .foreach(example => {
        val (loss, grad) = getLossAndGradient(example, params, options)
        val gradUpdate = params.computeGradient(grad, example)
        updateGradientSum(r0, r1, options,
          example, gradUpdate, gradSum)
        lossSum += loss
      })

    (lossSum, params.updatedParams(gradSum, options))
  }

  def updateGradientSum(r0: Double,
                        r1: Double,
                        options: TrainingOptions,
                        example: T,
                        updateGradients: Array[Double],
                        gradients: Array[Double]): Unit = {
    assert(updateGradients.length == gradients.length,
      s"${updateGradients.length}, ${gradients.length}")
    if (validGradient(gradients)) {
      val learningRate = getLearningRate(
        r0, r1, example, options)
      for (i <- gradients.indices) {
        gradients(i) += learningRate * updateGradients(i)
      }
    }
  }

  def validGradient(gradients: Array[Double]): Boolean = {
    valid(gradients)
  }

  private def valid(values: Array[Double]): Boolean = {
    for (value <- values) {
      if (value.isNaN) {
        return false
      }
    }
    true
  }

  def getLossAndGradient(example: T,
                         params: StrategyParams[T],
                         options: TrainingOptions): (Double, Double) = {
    // get loss and the gradient of loss w.r.t to model score
    val score = params.score(example)
    var absLoss = 0.0
    var grad = 0.0
    // optimal value should lie in [lowerBound, upperBound]
    val lowerBound = example.getLowerBound(options)
    val upperBound = example.getUpperBound(options)
    if (score < lowerBound) {
      absLoss += lowerBound - score
      grad -= 1.0
    } else if (score > upperBound) {
      absLoss += score - upperBound
      grad += 1.0
    } else {
      // no loss
    }

    if (options.maxAvgLossRatio > 0) {
      (absLoss / example.pivotValue, grad)
    } else {
      (absLoss, grad)
    }

  }

  protected def getOrDefaultParams(trainingOptions: TrainingOptions,
                         paramsOpt: Option[StrategyParams[T]]): StrategyParams[T] = {
    if (paramsOpt.isDefined) {
      paramsOpt.get
    } else {
      getDefaultParams(trainingOptions)
    }
  }
}

object BinaryTrainer {
  // TODO allow change this
  final val maxAvgLossSum = 1000
  private final val MIN_COUNT = 100
  def evalExample[T <: BinaryTrainingSample](
                  example: T,
                  params: StrategyParams[T],
                  ratio: Double): Double = {
    val prediction = params.score(example)
    example.lossRatioWithPrediction(prediction)
  }

  def getMetrics[T <: BinaryTrainingSample](evalExamples: Seq[T], params: StrategyParams[T]): BinaryMetrics = {
    val byKey = evalExamples.map(example => {
      val prediction = params.score(example)
      val predictionLower = example.predictionLower(prediction)
      val predictionIncrease = example.predictionIncrease(prediction)
      (example.label, predictionLower) -> predictionIncrease
    }
    )

    val r = byKey.groupBy(_._1).map {
      case (group, traversable) =>
        val len = traversable.length
        val sum = traversable.map(a => a._2).sum
        (group, (len, sum))
    }
    val regret = trueRegret(byKey, List(0.5, 0.75))
    BinaryMetrics.computeEvalMetricFromCounts(r, regret.head, regret(1))
  }

  def trueRegret(data: Seq[((Boolean,Boolean), Double)],
                 percentile: List[Double]): List[Double] = {
    val regret = data.filter{
      case((label, predictionLower), predictionIncrease) => label && predictionLower
    }.map(_._2)
    if (regret.nonEmpty) {
      percentile.map(p => {
        val pos = p * regret.length
        Sort.quickSelect(regret, pos.toInt)
      })
    } else {
      List.fill(percentile.length)(0)
    }
  }

  def enoughTrueSamples[T <: BinaryTrainingSample](options: TrainingOptions, examples: Seq[T]): Boolean = {
    val count = options.minTrueLabelCount
    if (count == 0) {
      true
    } else {
      examples.count(_.label) >= count
    }
  }

}
