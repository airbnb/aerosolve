package com.airbnb.common.ml.xgboost.model

import com.airbnb.common.ml.search.MonteCarloSearch
import com.airbnb.common.ml.util.{PipelineUtil, ScalaLogging}
import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost}

import scala.collection.mutable


class XGBoostModel(
    training: DMatrix,
    eval: DMatrix,
    val lossFunction: (Double, Double) => Double)
  extends MonteCarloSearch {

  override def eval(params: Map[String, Any]): Double = {
    XGBoostModel.eval(training, eval, params, lossFunction)
  }

  override def dispose(): Unit = {
    training.delete()
    eval.delete()
  }
}

object XGBoostModel extends ScalaLogging {

  def logLoss(p: Double, y: Double): Double = {
    -y * math.log(p) - (1 - y) * math.log(1 - p)
  }

  def linearLoss(p: Double, y: Double): Double = {
    math.abs((p - y)/y)
  }

  def eval(training: DMatrix,
      eval: DMatrix,
      params: Map[String, Any],
      lossFunction: (Double, Double) => Double
  ): Double = {
    val model = XGBoostModel.train(params, training)
    val prediction = model.predict(eval)

    val loss = eval.getLabel.zip(prediction).map(
      a => {
        math.abs(lossFunction(a._2(0), a._1))
      }
    ).sum / prediction.length
    model.dispose
    loss
  }

  def getModelByLabeledPoint(
      training: Seq[LabeledPoint],
      eval: Seq[LabeledPoint],
      loss: String
  ): XGBoostModel = {
    val trainingDMatrix = new DMatrix(training.iterator, null)
    val evalDMatrix = new DMatrix(eval.iterator, null)
    new XGBoostModel(trainingDMatrix, evalDMatrix, getLossFunction(loss))
  }

  def getModelByFile(
      training: String,
      eval: String,
      loss: String
  ): XGBoostModel = {
    getModelByFileWithLoss(training, eval, getLossFunction(loss))
  }

  private def getLossFunction(loss: String): (Double, Double) => Double = {
    val lossFun: (Double, Double) => Double = loss match {
      case "linear" => XGBoostModel.linearLoss
      case "logLoss" => XGBoostModel.logLoss
      case _ => throw new RuntimeException("unknown loss type")
    }
    lossFun
  }

  private def getModelByFileWithLoss(
      training: String,
      eval: String,
      loss: (Double, Double) => Double = logLoss
  ): XGBoostModel = {
    val trainingDMatrix = new DMatrix(training)
    val evalDMatrix = new DMatrix(eval)
    logger.info(s"trainingDMatrix ${trainingDMatrix.rowNum} ${evalDMatrix.rowNum}")

    new XGBoostModel(trainingDMatrix, evalDMatrix, loss)
  }

  def train(params: Map[String, Any], training: DMatrix): ml.dmlc.xgboost4j.scala.Booster = {
    val round = params("round").asInstanceOf[Int]
    XGBoost.train(training, params, round)
  }

  def trainAndSave(
      training: DMatrix,
      output: String,
      round: Int,
      paramMap: Map[String, Any]
  ): Unit = {
    // log train-error
    val watches = new mutable.HashMap[String, DMatrix]
    watches += "train" -> training
    logger.info(s"xgbtraining start $output")
    val model = XGBoost.train(training, paramMap, round, watches.toMap)
    // save model to the file.
    val stream = PipelineUtil.getHDFSOutputStream(output)
    model.saveModel(stream)
    model.dispose
    training.delete()
  }
}
