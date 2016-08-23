package com.airbnb.aerosolve.training.photon.ml.data

import java.io.File

import com.airbnb.aerosolve.core.{Example, FeatureVector}
import com.airbnb.aerosolve.core.transforms.Transformer
import com.airbnb.aerosolve.training.pipeline.GenericPipeline
import com.airbnb.aerosolve.training.photon.ml.data.PhotonMLUtils._
import com.typesafe.config.{Config, ConfigFactory, ConfigParseOptions, ConfigResolveOptions}
import org.apache.avro.generic.{GenericData, GenericRecord}
import org.apache.avro.mapred.AvroKey
import org.apache.avro.mapreduce.{AvroJob, AvroKeyOutputFormat}
import org.apache.avro.Schema
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.NullWritable
import org.apache.hadoop.mapreduce.Job
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import scala.collection.immutable
import scala.util.Try


/**
  * Providing a data conversion job casting Thrift [[com.airbnb.aerosolve.core.Example]] to PhotonML
  * Avro format: https://github.com/linkedin/photon-ml#input-data-format
  *
  * @see resources/demo_convert_example_to_photonml_avro.conf about how to set up job configuration.
  *
  */
object ConvertExamplesToPhotonMLAvroJob {
  private val logger = LoggerFactory.getLogger(this.getClass())
  private val LABEL = "label"

  /**
    * A single entry-point for launching this job from command line
    *
    * @param args Commandline arguments
    */
  def main(args: Array[String]): Unit = {
    val configPath = new File(args(0))

    val config = if (configPath.exists()) {
      ConfigFactory.load(ConfigFactory.parseFile(configPath))
    } else {
      ConfigFactory.load(configPath.getPath(),
        ConfigParseOptions.defaults.setAllowMissing(false),
        ConfigResolveOptions.defaults)
    }

    val sparkConf = new SparkConf().setAppName("Convert Thrift Example to PhotonML Avro")
    val sc = new SparkContext(sparkConf)

    convertExamplesToPhotonGAMEAvro(sc, config)
  }

  private def getLabelFromExample(example: Example): Double = {
    if (example.getExample().get(0).floatFeatures.get(GenericPipeline.LABEL).get("") > 0) {
      1d
    } else {
      0d
    }
  }

  /**
    * This method converts an RDD of examples to PhotonML GAME Avro formats and save to HDFS.
    * Could be used as an entry-point in programming.
    *
    * @param sc         The spark context
    * @param taskConfig config of the task
    */
  def convertExamplesToPhotonGAMEAvro(sc: SparkContext,
                                      taskConfig: Config): Unit = {
    val (transformedData, avroSchema) = loadExamples(sc, taskConfig)

    val outputPath = new Path(taskConfig.getString("output_path"))

    val fs = outputPath.getFileSystem(new Configuration())
    fs.delete(outputPath, true)

    val avroSchemaStr = avroSchema.toString()
    logger.info(s" Configured output Avro schema: ${avroSchema.toString(true)}")

    val metaDataExtractorClass = Try(taskConfig.getString("meta_data_extractor_class"))
      .getOrElse(classOf[SingleFloatFamilyMetaExtractor].getName())

    val metaDataExtractorContext = Try(taskConfig
      .getConfig("meta_data_extractor_context").entrySet().asScala
      .map { case entry => (entry.getKey(), entry.getValue().toString()) }.toMap
    ).getOrElse(immutable.Map[String, String]())

    val trainingExampleAvros = transformedData.mapPartitions { case iter =>
      val schema = new Schema.Parser().parse(avroSchemaStr)

      val metaDataExtractor = Class.forName(metaDataExtractorClass)
        .newInstance().asInstanceOf[ExampleMetaDataExtractor]

      iter.map { case example =>
        val record = new GenericData.Record(schema)

        val label = getLabelFromExample(example)

        // TODO decide which key to provide for training example, currently this is not a must-have field
        // record.put(AVRO_UID, key)
        record.put(AVRO_LABEL, label)
        record.put(AVRO_META_DATA_MAP,
          metaDataExtractor.buildMetaDataMap(example, metaDataExtractorContext))

        createAvroFeatureArrays(record, example, schema)

        record
      }
    }

    val job = Job.getInstance
    AvroJob.setOutputKeySchema(job, avroSchema)
    val hadoopConf = job.getConfiguration()

    // Compress Avro binary files
    hadoopConf.set("mapreduce.output.fileoutputformat.compress", "true")
    hadoopConf.set(AvroJob.CONF_OUTPUT_CODEC, "deflate")

    // Save as Avro records
    trainingExampleAvros
      .map { r => (new AvroKey(r), NullWritable.get()) }
      .saveAsNewAPIHadoopFile(outputPath.toString(),
        classOf[AvroKey[GenericRecord]],
        classOf[org.apache.hadoop.io.NullWritable],
        classOf[AvroKeyOutputFormat[GenericRecord]],
        hadoopConf)
  }

  // Load RDD[Example] according to context information
  private def loadExamples(sc: SparkContext,
                           taskConfig: Config): (RDD[Example], Schema) = {

    val hiveTraining = GenericPipeline.runQuery(sc, taskConfig.getString("input_hive_query"))
    val dataFrameSchema = hiveTraining.schema.fields

    val namespaces = (dataFrameSchema.map { case f =>
      val idx = f.name.indexOf("_")

      if (idx == -1) {
        f.name
      } else {
        f.name.substring(0, idx)
      }
    }.toSet[String] ++ taskConfig.getStringList("avro_extra_namespaces").asScala).toArray

    // label namespace should not be treated as a feature
    val avroSchema = makeTrainingExampleAvroSchema(featureNamespaces =
      namespaces.filter(_.toLowerCase != LABEL))

    // Read and cast to Examples
    val examples = hiveTraining
      .map(x => GenericPipeline.hiveTrainingToExample(x, dataFrameSchema, false))

    // Pass through transformers
    val transformer = new Transformer(taskConfig, "model_config")
    val transformerBC = sc.broadcast(transformer)

    val transformedExamples = examples.flatMap { case example =>
      example.example.asScala.map { fv: FeatureVector =>
        val newExample = new Example()
        newExample.setContext(example.context)
        newExample.addToExample(fv)
        transformerBC.value.combineContextAndItems(newExample)

        newExample
      }
    }

    (transformedExamples, avroSchema)
  }
}
