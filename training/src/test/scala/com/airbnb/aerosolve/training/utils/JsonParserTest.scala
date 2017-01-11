package com.airbnb.aerosolve.training.utils

import org.junit.Assert.assertTrue
import org.junit.Test
import org.slf4j.LoggerFactory

class JsonParserTest {
  val log = LoggerFactory.getLogger("JsonParserTest")

  val json = """
    { "timestamp":1410275590157,
      "host":"i-2589c30f",
      "data":{ "operation":"impression",
               "event_data":{
                "results":[
     { "hostingId":2485599,
     "feature_string":"1=4.0,2=2485599.0,3=0.7532774,4=0.0,5=0.32489797,6=12.0"
     },
     {"hostingId":1067885,
      "feature_string":"1=4.0,2=1067885.0,3=0.7684334,4=0.0"
     }]}
    }}
   """

  @Test def testParseAny = {
    val values = JsonParser.parseJson[Map[String, Any]](json);
    log.info(values.toString)
    assertTrue(values.get("timestamp") != None)
    val expected = 1410275590157L;
    assertTrue(values.get("timestamp").get.asInstanceOf[Long] ==
               expected)
    val data = values.get("data").get.asInstanceOf[Map[String, Any]]
    val eventData = data.get("event_data").get.asInstanceOf[Map[String, Any]]
    val resultOpt = eventData.get("results")
    log.info(eventData.toString)
    assertTrue(resultOpt != None)
    val results = resultOpt.get.asInstanceOf[List[Map[String, String]]]
    log.info(results.toString)
    val featureString : String = results.head.get("feature_string").get
    val features = featureString.split(",")
    val featuresMap = features
      .map(x => x.split("="))
      .map(x => (x(0).toInt, x(1).toDouble))
      .toMap
    log.info(featuresMap.toString)
    assertTrue(featuresMap.get(1).get > 3.9)
  }

}