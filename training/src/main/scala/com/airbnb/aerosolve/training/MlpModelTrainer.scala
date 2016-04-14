package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.features.{Family, Feature, FeatureRegistry, MultiFamilyVector}
import com.airbnb.aerosolve.core.models.MlpModel
import com.airbnb.aerosolve.core.util.FloatVector
import com.airbnb.aerosolve.core.{Example, FunctionForm}
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.util.Try

/**
  * A trainer that generates a MLP model.
  * TODO (Peng): add maxNorm regularizations
  */

object MlpModelTrainer {
  private final val log: Logger = LoggerFactory.getLogger("MlpModelTrainer")
  private final val LAYER_PREFIX : String = "$layer:"
  private final val NODE_PREFIX : String = "$node:"
  private final val BIAS_PREFIX : String = "$bias:"
  case class TrainerOptions(loss: String,
                            margin: Double, // margin in Hinge loss or epsilon in regression
                            iteration: Int, // number of iterations to run
                            subsample : Double, // determine mini-batch size
                            threshold : Double, // threshold for binary classification
                            labelFamily : Family,
                            learningRateInit : Double,  // initial learning rate
                            learningRateDecay : Double, // learning rate decay rate
                            momentumInit : Double, // initial momentum value
                            momentumEnd : Double,  // ending momentum value
                            momentumT : Int,
                            dropout : Double, // dropout rate
                            maxNorm : Double, // max norm
                            weightDecay : Double, // l2 regularization parameter
                            weightInitStd : Double, // weight initialization std
                            cache : String,
                            minCount : Int,
                            registry: FeatureRegistry
                           )

  case class NetWorkParams(activationFunctions: java.util.ArrayList[FunctionForm],
                           nodeNumber : java.util.ArrayList[Integer])

  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String,
            registry: FeatureRegistry) : MlpModel = {
    val trainerOptions = parseTrainingOptions(config.getConfig(key), registry)
    val networkOptions = parseNetworkOptions(config.getConfig(key))

    val raw : RDD[Example] =
      LinearRankerUtils
        .makePointwiseFloat(input, config, key, trainerOptions.registry)

    val pointwise = trainerOptions.cache match {
      case "memory" => raw.cache()
      case _ : String => raw
    }

    val model = setupModel(trainerOptions, networkOptions, pointwise)

    modelIteration(sc, trainerOptions, model, pointwise)

    trainerOptions.cache match {
      case "memory" => pointwise.unpersist()
      case _ => Unit
    }
    model
  }

  def modelIteration(sc : SparkContext,
                     options : TrainerOptions,
                     model : MlpModel,
                     pointwise : RDD[Example]) = {

    var learningRate = options.learningRateInit
    // initialize previous updates with 0
    val updateContainer = setupUpdateContainer(model)
    val N = Math.floor(1.0 / options.subsample).toInt

    for (iter <- 0 until options.iteration) {
      log.info(s"Iteration $iter")
      // -- update momentum
      val momentum = if (options.momentumT > 0) {
        updateMomentum(
          options.momentumInit,
          options.momentumEnd,
          options.momentumT,
          iter)
      } else {
        // if momentumT <= 0, we don't apply momentum based updating at all.
        0.0
      }

      for (k <- 0 until N) {
        val miniBatch = pointwise
          .sample(false, options.subsample)
        // compute gradients
        val gradientContainer = computeGradient(sc, options, model, miniBatch)
        // -- update all layer weights and bias and update container
        updateModel(model, gradientContainer, updateContainer,
          momentum.toFloat, learningRate.toFloat, options.dropout)

      }

      // update learning rate
      learningRate *= options.learningRateDecay
    }
  }

  def computeGradient(sc : SparkContext,
                      options : TrainerOptions,
                      model : MlpModel,
                      miniBatch : RDD[Example]) : Map[Feature, FloatVector] = {
    // compute the sum of gradient of examples in the mini-batch
    val modelBC = sc.broadcast(model)
    miniBatch
      .mapPartitions(partition => {
        val model = modelBC.value
        val gradient = mutable.HashMap[Feature, FloatVector]()
        partition.foreach(example => {
          val fv = example.only
          val score = if (options.dropout > 0) {
            model.forwardPropagationWithDropout(fv, options.dropout)
          } else {
            model.forwardPropagation(fv)
          }
          val grad = options.loss match {
            case "hinge" => computeHingeGradient(score, fv, options)
            case "regression" => computeRegressionGradient(score, fv, options)
            case _ => computeHingeGradient(score, fv, options)
          }
          // back-propagation for updating gradient
          // note: activations have been computed in forwardPropagation
          val outputLayerId = model.numHiddenLayers
          val func = model.activationFunction.get(outputLayerId)
          // delta: gradient of loss function w.r.t. node input
          // activation: the output of a node
          val outputNodeDelta = computeActivationGradient(score, func) * grad

          backPropagation(model, outputNodeDelta.toFloat, gradient, fv, options.weightDecay.toFloat)
        })
        gradient.iterator
      })
      .mapValues(fv => (fv, 1.0))
      .reduceByKey((a, b) => {
        a._1.add(b._1)
        (a._1, a._2 + b._2)
      })
      .mapValues(x => {
        x._1.scale(1.0f / x._2.toFloat)
        x._1
      })
      .collectAsMap()
      .toMap
  }

  def backPropagation(model: MlpModel,
                      outputNodeDelta: Float,
                      gradient: mutable.Map[Feature, FloatVector],
                      vector : MultiFamilyVector,
                      weightDecay: Double = 0.0d) = {
    // outputNodeDelta: gradient of the loss function w.r.t the input of the output node
    // set delta for the output layer
    var upperLayerDelta = new FloatVector(1)
    upperLayerDelta.set(0, outputNodeDelta)
    // compute gradient for bias at the output node
    val outputBiasGrad = new FloatVector(1)
    outputBiasGrad.set(0, outputNodeDelta)
    val outputBiasKey = vector.registry.feature(
      LAYER_PREFIX + model.numHiddenLayers.toString, BIAS_PREFIX)
    outputBiasGrad.add(gradient.getOrElse(outputBiasKey, new FloatVector(1)))
    gradient.put(outputBiasKey, outputBiasGrad)
    // update for hidden layers
    for (i <- (0 until model.numHiddenLayers).reverse) {
      // i decreases from numHiddenLayers-1 to 0
      val numNode = model.layerNodeNumber.get(i)
      val numNodeUpperLayer = model.layerNodeNumber.get(i + 1)
      val func = model.activationFunction.get(i)
      val thisLayerDelta = new FloatVector(numNode)
      // compute gradient of weights from the i-th layer to the (i+1)-th layer
      val activations = model.layerActivations.get(i)
      val hiddenLayerWeights = model.hiddenLayerWeights.get(i)
      val biasKey = vector.registry.feature(LAYER_PREFIX + i.toString, BIAS_PREFIX)
      val biasGrad = gradient.getOrElse(biasKey, new FloatVector(numNode))
      for (j <- 0 until numNode) {
        val key = vector.registry.feature(LAYER_PREFIX + i.toString, NODE_PREFIX + j.toString)
        val gradFv = gradient.getOrElse(key, new FloatVector(numNodeUpperLayer))
        gradFv.multiplyAdd(activations.get(j), upperLayerDelta)
        if (weightDecay > 0.0f) {
          val weight = model.hiddenLayerWeights.get(i).get(j)
          gradFv.multiplyAdd(weightDecay, weight)
        }
        gradient.put(key, gradFv)

        val grad = upperLayerDelta.dot(hiddenLayerWeights.get(j))
        val delta = computeActivationGradient(activations.get(j), func) * grad

        thisLayerDelta.set(j, delta.toFloat)
      }
      biasGrad.add(thisLayerDelta)
      if (weightDecay > 0.0d) {
        biasGrad.multiplyAdd(weightDecay, model.bias.get(i))
      }
      gradient.put(biasKey, biasGrad)
      upperLayerDelta = thisLayerDelta
    }

    // update for the input layer
    val numNodeUpperLayer = model.layerNodeNumber.get(0)
    for (fv <- vector.iterator) {
      val key = fv.feature
      // We only care about features in the model.
      if (model.inputLayerWeights.containsKey(key)) {
        val gradFv = gradient.getOrElse(key, new FloatVector(numNodeUpperLayer))
        gradFv.multiplyAdd(fv.value, upperLayerDelta)
        if (weightDecay > 0.0d) {
          val weight = model.inputLayerWeights.get(key)
          gradFv.multiplyAdd(weightDecay, weight)
        }
        gradient.put(key, gradFv)
      }
    }
  }

  def updateModel(model: MlpModel,
                  gradientContainer: Map[Feature, FloatVector],
                  updateContainer: mutable.Map[Feature, FloatVector],
                  momentum: Double,
                  learningRate: Double,
                  dropout: Double) = {
    // computing current updates based on previous updates and new gradient
    // then update model weights (also update the prevUpdateContainer)
    for ((key, prevUpdate) <- updateContainer) {
      val weightToUpdate : FloatVector = if (key.family.name.startsWith(LAYER_PREFIX)) {
        val layerId: Int = key.family.name.substring(LAYER_PREFIX.length).toInt
        assert(layerId >= 0 && layerId <= model.numHiddenLayers)
        if (key.name.equals(BIAS_PREFIX)) {
          // node bias updates
          model.bias.get(layerId)
        } else if (key.name.startsWith(NODE_PREFIX)) {
          val nodeId = key.name.substring(NODE_PREFIX.length).toInt
          // hidden layer weight updates
          model.hiddenLayerWeights.get(layerId).get(nodeId)
        } else {
          // error
          assert(false)
          new FloatVector()
        }
      } else {
        // input layer weight updates
        val inputLayerWeight = model.inputLayerWeights.get(key)
        if (inputLayerWeight != null) {
          inputLayerWeight
        } else {
          new FloatVector()
        }
      }
      if (weightToUpdate.length() > 0) {
        val gradient: FloatVector = gradientContainer.getOrElse(key, new FloatVector(weightToUpdate.length))
        val update: FloatVector = computeUpdates(prevUpdate, momentum, learningRate, gradient)
        // update the update container
        updateContainer.put(key, update)
        // update weights
        weightToUpdate.add(update)
      }
    }
  }

  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String,
                         registry: FeatureRegistry) = {
    val model = train(sc, input, config, key, registry)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }

  private def parseTrainingOptions(config : Config,
                                   registry: FeatureRegistry) : TrainerOptions = {
    TrainerOptions(
      loss = config.getString("loss"),
      margin = config.getDouble("margin"),
      iteration = config.getInt("iterations"),
      subsample = config.getDouble("subsample"),
      threshold = Try(config.getDouble("rank_threshold")).getOrElse(0.0),
      labelFamily = registry.family(config.getString("rank_key")),
      learningRateInit = config.getDouble("learning_rate_init"),
      learningRateDecay = Try(config.getDouble("learning_rate_decay")).getOrElse(1.0),
      momentumInit = Try(config.getDouble("momentum_init")).getOrElse(0.0),
      momentumEnd = Try(config.getDouble("momentum_end")).getOrElse(0.0),
      momentumT = Try(config.getInt("momentum_t")).getOrElse(0),
      dropout = Try(config.getDouble("dropout")).getOrElse(0.0),
      maxNorm = Try(config.getDouble("max_norm")).getOrElse(0.0),
      weightDecay = Try(config.getDouble("weight_decay")).getOrElse(0.0),
      weightInitStd = config.getDouble("weight_init_std"),
      cache = Try(config.getString("cache")).getOrElse(""),
      minCount = Try(config.getInt("min_count")).getOrElse(0),
      registry = registry
    )
  }

  private def parseNetworkOptions(config : Config) : NetWorkParams = {
    val activationStr = config.getStringList("activations")
    val activations = new java.util.ArrayList[FunctionForm]()
    for (func: String <- activationStr) {
      activations.append(getFunctionForm(func))
    }

    val nodeNumbers =  new java.util.ArrayList[Integer]()
    for (num : Integer <- config.getIntList("node_number")) {
      nodeNumbers.append(num)
    }

    NetWorkParams(
      activationFunctions = activations,
      nodeNumber = nodeNumbers
    )
  }

  def setupModel(trainerOptions : TrainerOptions,
                 networkOptions: NetWorkParams,
                 pointwise : RDD[Example]) : MlpModel = {
    val model = new MlpModel(
      networkOptions.activationFunctions,
      networkOptions.nodeNumber,
      trainerOptions.registry)

    val std = trainerOptions.weightInitStd.toFloat
    val stats = TrainingUtils.getFeatureStatistics(trainerOptions.minCount, pointwise)

    // set up input layer weights
    var count : Int = 0
    for ((feature, featureStats) <- stats) {
      if (feature.family != trainerOptions.labelFamily && !model.inputLayerWeights.containsKey(feature)) {
        count = count + 1
        model.inputLayerWeights.put(feature, FloatVector.getGaussianVector(model.layerNodeNumber.get(0), std))
      }
    }
    // set up hidden layer weights
    for (i <- 0 until model.numHiddenLayers) {
      val arr = new java.util.ArrayList[FloatVector]()
      for (j <- 0 until model.layerNodeNumber.get(i)) {
        val fv = FloatVector.getGaussianVector(model.layerNodeNumber.get(i + 1), std)
        arr.add(fv)
      }
      model.hiddenLayerWeights.put(i, arr)
    }
    // note: bias at each node initialized to zero in this trainer
    log.info(s"Total number of inputFeatures is $count")
    model
  }

  private def setupUpdateContainer(model: MlpModel) : mutable.Map[Feature, FloatVector] = {
    val container = mutable.HashMap[Feature, FloatVector]()
    // set up input layer weights gradient
    val n0 = model.layerNodeNumber.get(0)
    for (feature <- model.inputLayerWeights.keySet) {
      container.put(feature, new FloatVector(n0))
    }

    // set up hidden layer weights gradient
    val numHiddenLayers = model.numHiddenLayers
    for (i <- 0 until numHiddenLayers) {
      val thisLayerNodeNum = model.layerNodeNumber.get(i)
      val nextLayerNodeNum = model.layerNodeNumber.get(i + 1)
      for (j <- 0 until thisLayerNodeNum) {
        val key = model.registry.feature(LAYER_PREFIX + i.toString, NODE_PREFIX + j.toString)
        container.put(key, new FloatVector(nextLayerNodeNum))
      }
    }

    // set up bias gradient
    for (i <- 0 to numHiddenLayers) {
      // all bias in the same layer are put to the same FloatVector
      val key = model.registry.feature(LAYER_PREFIX + i.toString, BIAS_PREFIX)
      container.put(key, new FloatVector(model.layerNodeNumber.get(i)))
    }

    container
  }

  private def computeUpdates(prevUpdate: FloatVector,
                             momentum: Double,
                             learningRate: Double,
                             gradient: FloatVector): FloatVector = {
    // based on hinton's dropout paper: http://arxiv.org/pdf/1207.0580.pdf
    val update: FloatVector = new FloatVector(prevUpdate.length)
    update.multiplyAdd(momentum, prevUpdate)
    update.multiplyAdd(-(1.0f - momentum) * learningRate, gradient)
    update
  }

  private def getFunctionForm(func: String) : FunctionForm = {
    func match {
      case "sigmoid" => FunctionForm.SIGMOID
      case "relu" => FunctionForm.RELU
      case "tanh" => FunctionForm.TANH
      case "identity" => FunctionForm.IDENTITY
      case _ => assert(false); FunctionForm.SIGMOID
    }
  }

  private def updateMomentum(momentumInit: Double,
                             momentumEnd: Double,
                             momentumT: Int,
                             iter: Int) : Double = {
    if (iter >= momentumT)
      return momentumEnd
    val frac = iter.toDouble / momentumT
    frac * momentumInit + (1 - frac) * momentumEnd
  }

  private def computeHingeGradient(prediction: Double,
                                   fv: MultiFamilyVector,
                                   option: TrainerOptions): Double = {
    // Returns d_loss / d_output_activation
    // gradient of loss function w.r.t the output node activation
    val label = TrainingUtils.getLabel(fv, option.labelFamily, option.threshold)
    // loss = max(0.0, option.margin - label * prediction)
    if (option.margin - label * prediction > 0) {
      -label
    } else {
      0.0
    }
  }

  private def computeRegressionGradient(prediction: Double,
                                        fv: MultiFamilyVector,
                                        option: TrainerOptions): Double = {
    // epsilon-insensitive loss for regression (as in SVM regression)
    // loss = max(0.0, |prediction - label| - epsilon)
    // where epsilon = option.margin
    assert(option.margin > 0)
    val label = TrainingUtils.getLabel(fv, option.labelFamily)
    if (prediction - label > option.margin) {
      1.0
    } else if (prediction - label < - option.margin) {
      -1.0
    } else {
      0.0
    }
  }

  private def computeActivationGradient(activation: Double,
                                        func: FunctionForm): Double = {
    // compute the gradient of activation w.r.t input
    func match {
      case FunctionForm.SIGMOID => activation * (1.0 - activation)
      case FunctionForm.RELU => if (activation > 0) 1.0 else 0.0
      case FunctionForm.IDENTITY => 1.0
      case FunctionForm.TANH => 1.0 - activation * activation
    }
  }
}
