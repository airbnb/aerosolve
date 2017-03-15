package com.airbnb.aerosolve.training.strategy.trainer

import com.airbnb.aerosolve.training.strategy.config.TrainingOptions
import com.airbnb.aerosolve.training.strategy.data.BaseBinarySample
import com.airbnb.aerosolve.training.utils.TestUtil
import org.apache.spark.SparkContext
import org.apache.spark.sql.hive.HiveContext
import org.slf4j.LoggerFactory

object TrainerTestUtil {
  val log = LoggerFactory.getLogger("TrainerTestUtil")
  def getOptions(learning:String = "leadDays"): TrainingOptions = {
    TrainingOptions(
      // k1, k2
      0.98, 1.0,
      // lowerBound
      0.65, 1.03,
      // r0
      0.1, 0.2, 0.95,
      //numEpochs: Int, miniBatchSize: Int,
      100, 50,
      0.5,
      0, 0.0,
      //min: Array[Double], max: Array[Double],
      List(0.1, 0), List(3, 3),
      List(),
      //dsEval: String, learningRateType: String
      "2016-08-31", learning)
  }

  def getOptionsTanhv2: TrainingOptions = {
    TrainingOptions(
      // k1, k2
      0.98, 1.0,
      // lowerBound
      0.65, 1.03,
      // r0
      0.1, 0.2, 0.95,
      //numEpochs: Int, miniBatchSize: Int,
      100, 50,
      0.5,
      0, 0.5,
      //min: Array[Double], max: Array[Double],
      List(0.05, 4, -24),
      List(0.35, 25.0, -1.0),
      List(),
      //dsEval: String, learningRateType: String
      "2016-08-31", "")
  }

  def parseKey(cols: Array[String]): String = {
    cols(0)
  }

//  label: Boolean,
//  x: Double,
//  pivotValue: Double,
//  observedValue: Double
  def parseBaseData(cols: Array[String]): BaseBinarySample = {
    val label = cols(0).toBoolean
    BaseBinarySample(
      label,
      cols(1).toDouble,
      cols(2).toDouble,
      cols(3).toDouble)
  }

  def testSearchAllBestOptions(
       training: String,
       eval: String,
       hc: HiveContext,
       sc: SparkContext,
       trainer: BinaryTrainer[BaseBinarySample]):Unit = {
    val options = TrainerTestUtil.getOptionsTanhv2
    val trainingExamples = TestUtil.parseCSVToRDD(
      training, TrainerTestUtil.parseKey, TrainerTestUtil.parseBaseData, sc)
    val evalExamples = TestUtil.parseCSVToRDD(
      eval, TrainerTestUtil.parseKey, TrainerTestUtil.parseBaseData, sc)

    val r = trainer.
      searchBestOptionsPerModel(trainingExamples, evalExamples, Array(options)).collect()

    log.info(s"size ${r.length}")
    log.info(s"c ${r.head.toString}")
  }

}
