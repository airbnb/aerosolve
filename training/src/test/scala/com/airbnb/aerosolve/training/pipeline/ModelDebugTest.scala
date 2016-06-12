package com.airbnb.aerosolve.training.pipeline

import com.airbnb.aerosolve.core.{FunctionForm, ModelRecord}
import org.junit.Assert._
import org.junit.Test
import org.slf4j.LoggerFactory

class ModelDebugTest {
  val log = LoggerFactory.getLogger("ModelDebugTest")
  @Test
  def modelRecordToString : Unit = {
    val r: ModelRecord = new ModelRecord()
    r.setFeatureFamily("f")
    r.setFeatureName("n")
    r.setMaxVal(1)
    r.setMinVal(0)
    r.setWeightVector(java.util.Arrays.asList(1.0,2.0))
    r.setFunctionForm(FunctionForm.Point)

    assertEquals("f\u0001n\u00010.000000\u00011.000000\u0001[1.0, 2.0]\u0001\u00010.000000\u00010.000000\u00010.000000",
                 ModelDebug.modelRecordToString(r))
  }
}
