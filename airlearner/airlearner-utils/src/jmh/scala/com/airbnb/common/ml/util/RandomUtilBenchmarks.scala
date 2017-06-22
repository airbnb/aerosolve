package com.airbnb.common.ml.util

import scala.collection.mutable

import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.BenchmarkMode
import org.openjdk.jmh.annotations.Fork
import org.openjdk.jmh.annotations.Measurement
import org.openjdk.jmh.annotations.Mode
import org.openjdk.jmh.annotations.Param
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.State
import org.openjdk.jmh.annotations.Warmup


object RandomUtilBenchmarks {


  /**
   * Perform testing of different Scala collection types as input to the
   * `RandomUtil.sample` method.
   */
  @BenchmarkMode(Array(Mode.Throughput))
  @Fork(value = 1, warmups = 0)
  @Warmup(iterations = 4, time = 2)
  @Measurement(iterations = 8, time = 2)
  @State(value = Scope.Benchmark)
  class SampleBenchmarks {

    // Try different collection sizes
    @Param(value = Array("1000", "10000"))
    var numItemsToGenerate: Int = _

    val ratios: Seq[Double] = Seq(0.85, 0.1, 0.05)


    @Benchmark
    def sampleArray(): Seq[Seq[Int]] = {
      RandomUtil.sample(
        Array.range(1, numItemsToGenerate),
        ratios
      )
    }

    @Benchmark
    def sampleArraySeq(): Seq[Seq[Int]] = {
      RandomUtil.sample(
        mutable.ArraySeq.range(1, numItemsToGenerate),
        ratios
      )
    }

    @Benchmark
    def sampleList(): Seq[Seq[Int]] = {
      RandomUtil.sample(
        List.range(1, numItemsToGenerate),
        ratios
      )
    }

    @Benchmark
    def sampleSeq(): Seq[Seq[Int]] = {
      RandomUtil.sample(
        Seq.range(1, numItemsToGenerate),
        ratios
      )
    }

    @Benchmark
    def sampleVector(): Seq[Seq[Int]] = {
      RandomUtil.sample(
        Vector.range(1, numItemsToGenerate),
        ratios
      )
    }

    @Benchmark
    def sampleSeqToVector(): Seq[Seq[Int]] = {
      RandomUtil.sample(
        Seq.range(1, numItemsToGenerate).toVector,
        ratios
      )
    }
  }
}
