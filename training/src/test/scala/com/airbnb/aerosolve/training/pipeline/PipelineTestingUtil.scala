package com.airbnb.aerosolve.training.pipeline

import com.airbnb.aerosolve.core.{Example, LabelDictionaryEntry}
import com.airbnb.aerosolve.core.features.{FeatureRegistry, SimpleExample}
import com.airbnb.aerosolve.core.models.{FullRankLinearModel, LinearModel}
import com.airbnb.aerosolve.core.transforms.Transformer
import com.airbnb.aerosolve.core.util.FloatVector
import com.google.common.collect.ImmutableMap
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
  val registry = new FeatureRegistry

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

    new Transformer(ConfigFactory.parseString(config), "model_transforms", registry)
  }

  // Simple full rank linear model with 2 label classes and 2 features
  val fullRankLinearModel = {
    val model = new FullRankLinearModel(registry)

    model.labelToIndex(ImmutableMap.of("label1", 0, "label2", 1))

    val labelDictEntry1 = new LabelDictionaryEntry()
    labelDictEntry1.setLabel("label1")
    labelDictEntry1.setCount(50)

    val labelDictEntry2 = new LabelDictionaryEntry()
    labelDictEntry2.setLabel("label2")
    labelDictEntry2.setCount(100)

    val labelDictionary = new java.util.ArrayList[LabelDictionaryEntry]()

    labelDictionary.add(labelDictEntry1)
    labelDictionary.add(labelDictEntry2)

    model.labelDictionary(labelDictionary)

    val floatVector1 = new FloatVector(Array(1.2f, 2.1f))
    val floatVector2 = new FloatVector(Array(3.4f, -1.2f))

    model.weightVector.putAll(
      ImmutableMap.of(
        registry.feature("f", "feature1"), floatVector1,
        registry.feature("f", "feature2"), floatVector2)
    )

    model
  }

  // Simple linear model with 2 features
  val linearModel = {
    val model = new LinearModel(registry)

    model.weights.putAll(
      ImmutableMap.of(
        registry.feature("s", "feature1"), 1.4d,
        registry.feature("s", "feature2"), 1.3d))

    model
  }

  val multiclassExample1 = {
    val example = new SimpleExample(registry)
    val fv = example.createVector()

    fv.put("f", "feature1", 1.2)
    fv.put("f", "feature2", 5.6)
    fv.put("LABEL", "label1", 10.0)
    fv.put("LABEL", "label2", 9.0)

    example
  }

  val multiclassExample2: Example = {
    val example = new SimpleExample(registry)
    val fv = example.createVector()

    fv.put("f", "feature1", 1.8)
    fv.put("f", "feature2", -1.6)
    fv.put("LABEL", "label1", 8.0)
    fv.put("LABEL", "label2", 4.0)

    example
  }

  val linearExample1: Example = {
    val example = new SimpleExample(registry)
    val fv = example.createVector()

    fv.putString("s", "feature1")
    fv.putString("s", "feature2")
    fv.put("LABEL", "", 3.5)

    example
  }

  val linearExample2: Example = {
    val example = new SimpleExample(registry)
    val fv = example.createVector()

    fv.putString("s", "feature1")
    fv.put("LABEL", "", -2.0)

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
