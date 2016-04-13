package com.airbnb.aerosolve.training.pipeline

import java.io.{BufferedReader, BufferedWriter, InputStreamReader, OutputStreamWriter}
import java.net.URI
import java.util.GregorianCalendar

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.{Accumulator, SparkContext}
import org.joda.time.format.{DateTimeFormat, DateTimeFormatter}
import org.joda.time.{DateTime, Days}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object PipelineUtil {
  val log: Logger = LoggerFactory.getLogger("PipelineUtil")
  val hadoopConfiguration = new Configuration()

  def saveAndCommitAsTextFile[U](rdd: RDD[U], dest: String): Unit = {
    saveAndCommitAsTextFile(rdd, dest, 0)
  }

  def saveAndCommitAsTextFile[U](rdd: RDD[U], dest: String, overwrite: Boolean): Unit = {
    saveAndCommitAsTextFile(rdd, dest, 0, overwrite)
  }

  def saveAndCommitAsTextFile[U](rdd: RDD[U], dest: String, partition: Int,
                                 overwrite: Boolean = false): Unit = {
    log.info("Saving data to %s".format(dest))

    val hfs = FileSystem.get(
      new java.net.URI(dest), new Configuration())

    val tmpPath = new Path(dest + ".tmp")
    val destPath = new Path(dest)

    if (!hfs.exists(destPath) || overwrite) {
      try {
        if (hfs.exists(tmpPath)) {
          hfs.delete(tmpPath, true)
          log.info("deleted old tmp directory: " + tmpPath)
        }
        if (partition > 0) {
          // shuffle it to prevent imbalance output, which slows down the whole job.
          rdd.coalesce(partition, true)
            .saveAsTextFile(dest + ".tmp", classOf[GzipCodec])
        } else {
          rdd.saveAsTextFile(dest + ".tmp", classOf[GzipCodec])
        }
        if (hfs.exists(destPath)) {
          hfs.delete(destPath, true)
          log.info("deleted old directory: " + destPath)
        }
        log.info("committing " + dest)
        hfs.rename(tmpPath, destPath)
        log.info("committed " + dest)
      } catch {
        case e: Exception => {
          log.error("exception during save: ", e)
          log.info("deleting failed data for " + dest + ".tmp")
          hfs.delete(tmpPath, true)
          log.info("deleted " + dest + ".tmp")
          throw e
        }
      }
    } else {
      log.info("data already exists for " + dest)
    }
  }

  def getLastPartition(hc: HiveContext, table: String): String = {
    val query = "SHOW PARTITIONS %s".format(table)
    val partitions = hc.sql(query).collect.map(_.getString(0)).sorted
    partitions
      .last /* e.g. ds=2015-01-01 */
      .split("=")
      .apply(1)
  }

  def ts(year: Int, month: Int, day: Int) : Long = {
    val date = new GregorianCalendar(year, month - 1, day)
    (date.getTimeInMillis()/1000)/86400
  }

  def ts(ds: String): Long  = {
    ts(ds.substring(0,4).toInt, ds.substring(5,7).toInt, ds.substring(8,10).toInt)
  }

  // Returns a date iterator from start to end date.
  def dateRange(from: String, to: String): Seq[DateTime] = {
    val formatter : DateTimeFormatter = DateTimeFormat.forPattern("yyyy-MM-dd")
    val startDate = formatter.parseDateTime(from)
    val endDate = formatter.parseDateTime(to)
    val numberOfDays = Days.daysBetween(startDate, endDate).getDays()

    for(n <- 0 to numberOfDays) yield startDate.plusDays(n)
  }

  // Returns a date iterator from today - from to today = to
  def dateRangeFromToday(from : Int, to : Int): Seq[DateTime] = {
    val now = new DateTime()
    val today = now.toLocalDate()

    val startDate = now.plusDays(from)
    val endDate = now.plusDays(to)
    val numberOfDays = Days.daysBetween(startDate, endDate).getDays()

    for (n <- 0 to numberOfDays) yield startDate.plusDays(n)
  }

  def dateDiff(from: String, to: String): Int = {
    val formatter : DateTimeFormatter = DateTimeFormat.forPattern("yyyy-MM-dd")
    val startDate = formatter.parseDateTime(from)
    val endDate = formatter.parseDateTime(to)
    val numberOfDays = Days.daysBetween(startDate, endDate).getDays()
    numberOfDays
  }

  def dateMinus(date: String, days: Int): String = {
    // return a string for the date which is 'days' earlier than 'date'
    // e.g. dateMinus("2015-06-01", 1) returns "2015-05-31"
    val formatter : DateTimeFormatter = DateTimeFormat.forPattern("yyyy-MM-dd")
    val dateFmt = formatter.parseDateTime(date)
    formatter.print(dateFmt.minusDays(days))
  }

  def datePlus(date: String, days: Int): String = {
    // return a string for the date which is 'days' earlier than 'date'
    // e.g. datePlus("2015-06-01", 1) returns "2015-06-02"
    dateMinus(date, -days)
  }

  // Returns a date range starting numDays before and ending at the desired date
  def dateRangeUntil(to: String, numDays: Int): Seq[DateTime] = {
    val formatter : DateTimeFormatter = DateTimeFormat.forPattern("yyyy-MM-dd")
    val endDate = formatter.parseDateTime(to)

    for(n <- numDays to 0 by -1) yield endDate.minusDays(n)
  }

  def dayOfYear(date: String): Int = {
    // return day of year of a date
    // e.g. dayOfYear("2015-01-01") returns 1
    val formatter : DateTimeFormatter = DateTimeFormat.forPattern("yyyy-MM-dd")
    val dateFmt = formatter.parseDateTime(date)
    dateFmt.dayOfYear().get()
  }

  def dayOfWeek(date: String): Int = {
    // return day of week of a date
    // e.g. if date is Monday, returns 1,
    // if date is Sunday, returns 0 (this is to be consistent with the pricing training data)
    val formatter : DateTimeFormatter = DateTimeFormat.forPattern("yyyy-MM-dd")
    val dateFmt = formatter.parseDateTime(date)
    dateFmt.dayOfWeek().get() % 7
  }

  def deleteHDFSFiles(path: String) = {
    log.info("Deleting files at " + path)
    val hfs = FileSystem.get(new Configuration())
    hfs.delete(new Path(path), true)
    log.info("Deleted.")
  }

  def rowIsNotNull(row : Row, count : Int) : Boolean = {
    for (i <- 0 until count) {
      if (row.isNullAt(i)) return false
    }
    return true
  }

  // Create a pair of spark accumulators that track (success, failure) counts.
  def addStatusAccumulators(sc: SparkContext,
                            accName: String,
                            accumulators: mutable.Map[String, Accumulator[Int]]): Unit = {
    accumulators.put(accName + ".success", sc.accumulator(0, accName + ".success"))
    accumulators.put(accName + ".failure", sc.accumulator(0, accName + ".failure"))
  }


  def countAllFailureCounters(accumulators: mutable.Map[String, Accumulator[Int]]): Long = {
    var failureCount = 0
    for ( (name, accumulator) <- accumulators) {
      log.info("- Accumulator {} : {}", name, accumulator.value )
      if (name.endsWith(".failure")) {
        failureCount += accumulator.value
      }
    }
    failureCount
  }

  def validateSuccessCounters(accumulators: mutable.Map[String, Accumulator[Int]],
                              minSuccess: Int): Boolean = {
    for ( (name, accumulator) <- accumulators) {
      if (name.endsWith(".success") && accumulator.value < minSuccess) {
        log.error("Failed counter: {} = {} < {}", name, accumulator.value.toString, minSuccess.toString)
        return false
      }
    }
    true
  }

  // TODO(kim): cleanup, write to hdfs directly instead of via SparkContext.
  def saveCountersAsTextFile(accumulators: mutable.Map[String, Accumulator[Int]],
                             sc: SparkContext,
                             hdfsFilePath: String): Unit = {
    var summary = Array("Summarizing counters:")

    for ( (name, accumulator) <- accumulators) {
      val logLine = "- %s = %d".format(name, accumulator.value )
      summary :+= logLine
      log.info(logLine)
    }

    saveAndCommitAsTextFile(sc.parallelize(summary), hdfsFilePath, 1, true)
  }

  def writeStringToFile(str: String, output: String) = {
    val fs = FileSystem.get(new URI(output), hadoopConfiguration)
    val path = new Path(output)
    val stream = fs.create(path, true)
    val writer = new BufferedWriter(new OutputStreamWriter(stream))
    writer.write(str)
    writer.close()
  }

  def readStringFromFile(inputFile : String): String = {
    val fs = FileSystem.get(new URI(inputFile), hadoopConfiguration)
    val path = new Path(inputFile)
    val stream = fs.open(path)
    val reader = new BufferedReader(new InputStreamReader(stream))
    val str = Stream.continually(reader.readLine()).takeWhile(_ != null).mkString("\n")
    str
  }

  def averageByKey[U:ClassTag](input: RDD[(U, Double)]) : RDD[(U, Double)] = {
    input
      .mapValues(value => (value, 1.0))
      .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))
      .mapValues(x => x._1 / x._2)
  }

  def copyFiles(srcPath: String, destPath: String, deleteSource: Boolean = false) = {
    val src = new Path(srcPath)
    val dest = new Path(destPath)
    val fsConfig = new Configuration()
    val fs = FileSystem.get(new java.net.URI(destPath), fsConfig)
    log.info("Copying successful from " + src + " to " + dest)
    try {
      FileUtil.copy(fs, src, fs, dest, deleteSource, fsConfig)
    } catch {
      case e: Exception => {
        log.info("Copy failed " + dest)
        System.exit(-1)
      }
    }
    log.info("Copy done.")
  }

  def getAllDirOrFiles(dirPath: String) : Array[String] = {
    // get all files under a directory
    val fsConfig = new Configuration()
    val fs = FileSystem.get(new java.net.URI(dirPath), fsConfig)
    val allFiles = fs.listStatus(new Path(dirPath))
    val out = new ArrayBuffer[String]()
    allFiles.foreach(f =>
      out.append(f.getPath.getName)
    )
    out.toArray
  }
}
