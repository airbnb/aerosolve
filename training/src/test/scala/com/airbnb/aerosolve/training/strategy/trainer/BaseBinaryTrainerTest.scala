package com.airbnb.aerosolve.training.strategy.trainer

import com.airbnb.aerosolve.training.strategy.data.BaseBinarySample
import com.airbnb.aerosolve.training.strategy.params.BaseParam
import com.airbnb.aerosolve.training.utils.TestWithHiveContext
import org.junit.Test
import org.slf4j.LoggerFactory

class BaseBinaryTrainerTest extends TestWithHiveContext {
  val log = LoggerFactory.getLogger("BaseBinaryTrainerTest")

  @Test
  def searchAllBestOptions():Unit = {
    val trainer = BaseBinaryTrainer(BaseParam(), BaseBinarySample)
    TrainerTestUtil.testSearchAllBestOptions("/train.csv", "/eval.csv", hc, sc, trainer)
  }
}
