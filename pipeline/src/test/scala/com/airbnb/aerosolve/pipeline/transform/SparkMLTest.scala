package com.airbnb.aerosolve.pipeline.transform

import com.airbnb.aerosolve.core.{Example, FeatureVector}
import com.airbnb.aerosolve.pipeline.transformers.RowTransformer
import com.airbnb.aerosolve.pipeline.transformers.continuous.Scaler
import com.airbnb.aerosolve.pipeline.transformers.produce.FamilyProducer
import com.airbnb.aerosolve.pipeline.transformers.select.FamilySelector
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.{AfterClass, BeforeClass, Test}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import com.airbnb.aerosolve.pipeline.SparkMLSupport._
import com.airbnb.aerosolve.pipeline.AerosolveSupport._

/**
 *
 */
class SparkMLTest {

  var sc: SQLContext = _

  val pipeline = RowTransformer(FamilySelector[Double]("loc"), FamilyProducer[Double]("out"))
    .map(
      Scaler(3.0)
        .andThen(Scaler(1.5))
        .andThen(Scaler(4.2)))


  val forest = new RandomForestClassifier()
    .setFeaturesCol("features")
    .setLabelCol("labels")

  @BeforeClass
  def setup() = {
    val sparkConf = new SparkConf()
      .set("spark.default.parallelism", "1")
      .set("spark.sql.shuffle.partitions", "1")
      .set("spark.ui.enabled", "false")
    val context = new SparkContext("local", "Model.Test", sparkConf)
    sc = new SQLContext(context)
    val tests = Seq(
      TestData(25, "programmer", 22.45, -33.21, 1.0),
      TestData(36, "lawyer", -3.45, -74.12, 0.0),
      TestData(19, "student", 2.12, 12.01, 0.0))
    sc.createDataFrame(tests).registerTempTable("test_data")
  }

  @AfterClass def teardown(): Unit = {
    sc.sparkContext.stop()
  }

  @Test
  def testRandomForestFromExamples: Unit = {
    val (examples, label, numPos) = makeClassificationExamples
    val input = sc.sparkContext
      .parallelize(examples)
      .flatMap(_.example.asScala.map(_.toRow))
    val data: DataFrame = pipeline(input)
    //val better = pipeline.andThen(forest.fit(data))
    /*val vectors: RDD[AerosolveVector] = input.flatMap(_.getExample.asScala
      .map(x => AerosolveTrainingVector(x)))
    val better = pipeline.apply(vectors)
    val df = DataFramer("$rank", sc).apply(better)
    val model = forest.fit(df)
    model.save("forest.model")
    val otherModel = RandomForests.load("forest.model")
    val scoring = pipeline.andThen(otherModel) */
  }

  @Test
  def testRandomForestFromDataFrame: Unit = {
    /*val df = sc.sql("select * from test_data")
    val better = pipeline.apply(df)
    val mode*/
  }

  def makeClassificationExamples = {
    val examples = ArrayBuffer[Example]()
    val label = ArrayBuffer[Double]()
    val rnd = new java.util.Random(1234)
    var numPos : Int = 0
    for (i <- 0 until 500) {
      val x = 2.0 * rnd.nextDouble() - 1.0
      val y = 10.0 * (2.0 * rnd.nextDouble() - 1.0)
      val poly = x * x + 0.1 * y * y + 0.1 * x + 0.2 * y - 0.1 + Math.sin(x)
      val rank = if (poly < 1.0) {
        1.0
      } else {
        -1.0
      }
      if (rank > 0) numPos = numPos + 1
      label += rank
      examples += makeExample(x, y, rank)
    }
    (examples, label, numPos)
  }

  def makeExample(x : Double,
                  y : Double,
                  target : Double) : Example = {
    val example = new Example
    val item: FeatureVector = new FeatureVector
    item.setFloatFeatures(new java.util.HashMap)
    item.setStringFeatures(new java.util.HashMap)
    val floatFeatures = item.getFloatFeatures
    val stringFeatures = item.getStringFeatures
    // A string feature that is always on.
    stringFeatures.put("BIAS", new java.util.HashSet)
    stringFeatures.get("BIAS").add("B")
    // A string feature that is sometimes on
    if (x + y < 0) {
      stringFeatures.put("NEG", new java.util.HashSet)
      stringFeatures.get("NEG").add("T")
    }
    floatFeatures.put("$rank", new java.util.HashMap)
    floatFeatures.get("$rank").put("", target)
    floatFeatures.put("loc", new java.util.HashMap)
    val loc = floatFeatures.get("loc")
    loc.put("x", x)
    loc.put("y", y)
    example.addToExample(item)
    example
  }

  case class TestData(age: Int, occupation: String, lat: Double, long: Double, label: Double)
}
