package com.airbnb.aerosolve.training.strategy.config

import com.typesafe.config.{Config, ConfigFactory}
import org.junit.Assert.assertEquals
import org.junit.Test
import org.slf4j.LoggerFactory

class SearchParamsTest {
  val log = LoggerFactory.getLogger("SearchParamsTest")
  def makeDoubleConfig: Config = {
    val config = """
      |param_search {
      |  params: ["lower_bound", "upper_bound"]
      |  lower_bound: [0.65, 0.7, 0.75]
      |  upper_bound: [1.03, 1.05, 1.08]
      |  list_params: ["min", "max"]
      |  min: [{list:[0.0, 1, -4]}, {list:[0.1, 2, -5]}]
      |  max: [{list:[2.0, 18, 4]}, {list:[3, 20, 4]}]
      |}
    """.stripMargin
    ConfigFactory.parseString(config)
  }

  def makeSingleConfig: Config = {
    val config = """
                   |param_search {
                   |  params: ["lower_bound"]
                   |  lower_bound: [0.65, 0.7, 0.75]
                   |  upper_bound: [1.03, 1.05, 1.08]
                   |  list_params: ["min", "max"]
                   |  min: [{list:[0.0, 1, -4]}, {list:[0.1, 2, -5]}]
                   |  max: [{list:[2.0, 18, 4]}, {list:[3, 20, 4]}]
                   |}
                 """.stripMargin
    ConfigFactory.parseString(config)
  }

  @Test
  def testParseDouble(): Unit = {
    val config = makeDoubleConfig
    val p1 = SearchParams.loadDoubleFromConfig(config.getConfig("param_search"))

    assertEquals(9, p1.paramCombinations.length)
    p1.paramCombinations.map(p => {
      val str = SearchParams.prettyPrint(p1.paramNames, p)
      log.info(s"p: $str")
    }
    )
  }

  @Test
  def testParseSingle(): Unit = {
    val config = makeSingleConfig
    val p1 = SearchParams.loadDoubleFromConfig(config.getConfig("param_search"))

    assertEquals(3, p1.paramCombinations.length)
    p1.paramCombinations.map(p => {
      val str = SearchParams.prettyPrint(p1.paramNames, p)
      log.info(s"p: $str")
    }
    )
  }

  @Test
  def testParseList(): Unit = {
    val config = makeDoubleConfig
    val p1 = SearchParams.loadListDoubleFromConfig(config.getConfig("param_search"))

    assertEquals(4, p1.paramCombinations.length)
    p1.paramCombinations.map(p => {
      val str = SearchParams.prettyPrint(p1.paramNames, p)
      log.info(s"p: $str")
    }
    )
  }
}
