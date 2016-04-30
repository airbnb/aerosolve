package com.airbnb.aerosolve.training.pipeline

import com.airbnb.aerosolve.core.{Example, FeatureVector, LabelDictionaryEntry}
import com.airbnb.aerosolve.core.models.{FullRankLinearModel, LinearModel}
import com.airbnb.aerosolve.core.transforms.Transformer
import com.airbnb.aerosolve.core.util.FloatVector
import com.google.common.collect.{ImmutableMap, ImmutableSet}
import com.typesafe.config.ConfigFactory
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.mockito.Matchers._
import org.mockito.Mockito._

import scala.language.implicitConversions
import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

/*
 * Misc. utilities that may be useful for testing Spark pipelines.
 */
object PipelineTestingUtil {

  val transformer = {
    val config = """
                   |identity_transform {
                   |  transform : list
                   |  transforms : [ ]
                   |}
                   |
                   |model_transforms {
                   |  context_transform : identity_transform
                   |  item_transform : identity_transform
                   |  combined_transform : identity_transform
                   |}
                 """.stripMargin

    new Transformer(ConfigFactory.parseString(config), "model_transforms")
  }

  // Simple full rank linear model with 2 label classes and 2 features
  val fullRankLinearModel = {
    val model = new FullRankLinearModel()

    model.setLabelToIndex(ImmutableMap.of("label1", 0, "label2", 1))

    val labelDictEntry1 = new LabelDictionaryEntry()
    labelDictEntry1.setLabel("label1")
    labelDictEntry1.setCount(50)

    val labelDictEntry2 = new LabelDictionaryEntry()
    labelDictEntry2.setLabel("label2")
    labelDictEntry2.setCount(100)

    val labelDictionary = new java.util.ArrayList[LabelDictionaryEntry]()

    labelDictionary.add(labelDictEntry1)
    labelDictionary.add(labelDictEntry2)

    model.setLabelDictionary(labelDictionary)

    val floatVector1 = new FloatVector(Array(1.2f, 2.1f))
    val floatVector2 = new FloatVector(Array(3.4f, -1.2f))

    model.setWeightVector(
      ImmutableMap.of(
        "f", ImmutableMap.of("feature1", floatVector1, "feature2", floatVector2)
      )
    )

    model
  }

  // Simple linear model with 2 features
  val linearModel = {
    val model = new LinearModel()

    model.setWeights(ImmutableMap.of("s", ImmutableMap.of("feature1", 1.4f, "feature2", 1.3f)))

    model
  }

  val multiclassExample1 = {
    val example = new Example()
    val fv = new FeatureVector()

    fv.setFloatFeatures(ImmutableMap.of(
      "f", ImmutableMap.of("feature1", 1.2, "feature2", 5.6),
      "LABEL", ImmutableMap.of("label1", 10.0, "label2", 9.0)
    ))

    example.addToExample(fv)

    example
  }

  val multiclassExample2 = {
    val example = new Example()
    val fv = new FeatureVector()

    fv.setFloatFeatures(ImmutableMap.of(
      "f", ImmutableMap.of("feature1", 1.8, "feature2", -1.6),
      "LABEL", ImmutableMap.of("label1", 8.0, "label2", 4.0)
    ))

    example.addToExample(fv)

    example
  }

  val linearExample1 = {
    val example = new Example()
    val fv = new FeatureVector()

    fv.setFloatFeatures(ImmutableMap.of(
      "LABEL", ImmutableMap.of("", 3.5)
    ))

    fv.setStringFeatures(ImmutableMap.of(
      "s", ImmutableSet.of("feature1", "feature2")
    ))

    example.addToExample(fv)

    example
  }

  val linearExample2 = {
    val example = new Example()
    val fv = new FeatureVector()

    fv.setFloatFeatures(ImmutableMap.of(
      "LABEL", ImmutableMap.of("", -2.0)
    ))

    fv.setStringFeatures(ImmutableMap.of(
      "s", ImmutableSet.of("feature1")
    ))

    example.addToExample(fv)

    example
  }

  def generateSparkContext = {
    val sparkConf =
      new SparkConf()
        .setMaster("local[2]")
        .setAppName("PipelineTestingUtil")
        .set("spark.io.compression.codec", "lz4")

    new SparkContext(sparkConf)
  }

  /*
   * Wrapper that generates a local SparkContext for a test and then
   * cleans everything up after test completion.
   */
  def withSparkContext[B](f: SparkContext => B): B = {
    val sc = generateSparkContext

    try {
      f(sc)
    } finally {
      sc.stop
      System.clearProperty("spark.master.port")
    }
  }

  /*
   * Create a mock HiveContext that responds to sql calls by returning
   * the argument results in a SchemaRDD.
   *
   * Works for any case class.
   */
  def createFakeHiveContext[A <: Product: TypeTag : ClassTag](
      sc: SparkContext, results: Seq[A]): HiveContext = {
    val mockHiveContext = mock(classOf[HiveContext])

    val sqlContext = new SQLContext(sc)

    when(mockHiveContext.sql(anyString())).thenReturn(sqlContext.createDataFrame(results))

    mockHiveContext
  }

  /*
   * Create a fake row and schema for a given case class.
   */
  def createFakeRowAndSchema[A <: Product: TypeTag : ClassTag](
      sc: SparkContext, result: A): (Row, StructType) = {
    val sqlContext = new SQLContext(sc)

    val sqlResult = sqlContext.createDataFrame(Seq(result))

    (sqlResult.head(), sqlResult.schema)
  }
}
