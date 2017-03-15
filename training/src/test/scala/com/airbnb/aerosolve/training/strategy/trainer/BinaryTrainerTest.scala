package com.airbnb.aerosolve.training.strategy.trainer

import com.airbnb.aerosolve.training.strategy.data.BinarySampleTest
import com.airbnb.aerosolve.training.strategy.eval.BinaryMetrics
import com.airbnb.aerosolve.training.strategy.params.BaseParam
import org.junit.Test
import org.slf4j.LoggerFactory

class BinaryTrainerTest {
  val log = LoggerFactory.getLogger("BinaryTrainerTest")

   @Test
  def evalExample():Unit = {
    val params = BaseParam.getDefault
    val examples = BinarySampleTest.getExamples
    examples.map(e => {
      val score = params.score(e)
      val result = BinaryTrainer.evalExample(e, params, 0.5)
      log.info(s"score $score value ${e.observedValue} result $result")
    })
  }

  @Test
  def getMetrics():Unit = {
    val params = BaseParam.getDefault
    val examples = BinarySampleTest.getExamplesSeq
    val m = BinaryTrainer.getMetrics(examples, params)

    log.debug(s"metrics ${BinaryMetrics.evalMetricsHeader}")
    log.debug(s"metrics ${m.toTSVRow}")
  }
}
