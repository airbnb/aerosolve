package com.airbnb.aerosolve.core.models;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.*;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.util.Util;
import com.airbnb.aerosolve.core.util.FloatVector;
import lombok.Getter;
import lombok.Setter;


/**
 * Multilayer perceptron (MLP) model https://en.wikipedia.org/wiki/Multilayer_perceptron
 * The current implementation is for the case where there is only one output node.
 */
public class MlpModel extends AbstractModel {

  private static final long serialVersionUID = -6870862764598907090L;

  // weights that define the projection from input layer (layer0) to the first hidden layer
  // or the output layer
  @Getter
  @Setter
  private Map<String, Map<String, FloatVector>> inputLayerWeights;

  // if there is hidden layer, this defines the projection
  // from one hidden layer to the next hidden layer or output layer
  // map from hidden layer id to hidden layer weights, id starts from 0
  @Getter
  @Setter
  private Map<Integer, ArrayList<FloatVector>> hiddenLayerWeights;

  // map from layer Id to bias applied on each node in the layer
  @Getter
  @Setter
  private Map<Integer, FloatVector> bias;

  @Getter
  @Setter
  private ArrayList<FunctionForm> activationFunction;

  // number of layers (excluding input layer and output layer)
  @Getter
  @Setter
  private int numHiddenLayers;

  // number of nodes for each hidden layer and output layer (does not include input layer)
  @Getter
  @Setter
  private ArrayList<Integer> layerNodeNumber;

  @Getter
  @Setter
  private Map<Integer, FloatVector> layerActivations;

  public MlpModel() {
    layerNodeNumber = new ArrayList<>();
    inputLayerWeights = new HashMap<>();
    hiddenLayerWeights = new HashMap<>();
    layerActivations = new HashMap<>();
    bias = new HashMap<>();
    activationFunction = new ArrayList<>();
  }

  public MlpModel(ArrayList<FunctionForm> activation, ArrayList<Integer> nodeNumbers) {
    // n is the number of hidden layers (including output layer, excluding input layer)
    // activation specifies activation function
    // nodeNumbers: specifies number of nodes in each hidden layer
    numHiddenLayers = nodeNumbers.size() - 1; // excluding output layer
    activationFunction = activation;
    layerNodeNumber = nodeNumbers;
    assert(activation.size() == numHiddenLayers + 1);
    inputLayerWeights = new HashMap<>();
    hiddenLayerWeights = new HashMap<>();
    // bias including the bias added at the output layer
    bias = new HashMap<>();
    layerActivations = new HashMap<>();

    for (int i = 0; i <= numHiddenLayers; i++) {
      int nodeNum = nodeNumbers.get(i);
      if (i < numHiddenLayers) {
        hiddenLayerWeights.put(i, new ArrayList<>(nodeNum));
      }
      bias.put(i, new FloatVector(nodeNum));
      layerActivations.put(i, new FloatVector(nodeNum));
    }
  }

  @Override
  public float scoreItem(FeatureVector combinedItem) {
    Map<String, Map<String, Double>> flatFeatures = Util.flattenFeature(combinedItem);
    return forwardPropagation(flatFeatures);
  }

  public float forwardPropagation(Map<String, Map<String, Double>> flatFeatures) {
    projectInputLayer(flatFeatures, 0.0);
    for (int i = 0; i < numHiddenLayers; i++) {
      projectHiddenLayer(i, 0.0);
    }
    return layerActivations.get(numHiddenLayers).get(0);
  }

  public float forwardPropagationWithDropout(Map<String, Map<String, Double>> flatFeatures, Double dropout) {
    // reference: George E. Dahl et al. "IMPROVING DEEP NEURAL NETWORKS FOR LVCSR USING RECTIFIED LINEAR UNITS AND DROPOUT"
    // scale the input to a node by 1/(1-dropout), so that we don't need to rescale model weights after training
    // make sure the value is between 0 and 1
    assert(dropout > 0.0);
    assert(dropout < 1.0);
    projectInputLayer(flatFeatures, dropout);
    for (int i = 0; i < numHiddenLayers; i++) {
      projectHiddenLayer(i, dropout);
    }
    return layerActivations.get(numHiddenLayers).get(0);
  }

  public FloatVector projectInputLayer(Map<String, Map<String, Double>> flatFeatures, Double dropout) {
    // compute the projection from input feature space to the first hidden layer or
    // output layer if there is no hidden layer
    // output: fvProjection is a float vector representing the activation at the first layer after input layer
    int outputNodeNum = layerNodeNumber.get(0);
    FloatVector fvProjection = layerActivations.get(0);
    if (fvProjection == null) {
      fvProjection = new FloatVector(outputNodeNum);
      layerActivations.put(0, fvProjection);
    } else {
      // recompute activation every time we do forward propagation
      fvProjection.setConstant(0.0f);
    }

    for (Map.Entry<String, Map<String, Double>> entry : flatFeatures.entrySet()) {
      Map<String, FloatVector> family = inputLayerWeights.get(entry.getKey());
      if (family != null) {
        for (Map.Entry<String, Double> feature : entry.getValue().entrySet()) {
          FloatVector vec = family.get(feature.getKey());
          if (vec != null) {
            if (dropout > 0.0 && Math.random() < dropout) continue;
            fvProjection.multiplyAdd(feature.getValue().floatValue(), vec);
          }
        }
      }
    }
    if (dropout > 0.0 && dropout < 1.0) {
      fvProjection.scale(1.0f / (1.0f - dropout.floatValue()));
    }
    // add bias for the first hidden layer or output layer
    fvProjection.add(bias.get(0));
    applyActivation(fvProjection, activationFunction.get(0));
    return fvProjection;
  }

  public FloatVector projectHiddenLayer(int hiddenLayerId, Double dropout) {
    int outputLayerId = hiddenLayerId + 1;
    int outputDim = layerNodeNumber.get(outputLayerId);
    FloatVector output = layerActivations.get(outputLayerId);
    if (output == null) {
      output = new FloatVector(outputDim);
      layerActivations.put(outputLayerId, output);
    } else {
      output.setConstant(0.0f);
    }

    FloatVector input = layerActivations.get(hiddenLayerId);
    ArrayList<FloatVector> weights = hiddenLayerWeights.get(hiddenLayerId);
    for (int i = 0; i < input.length(); i++) {
      if (dropout > 0.0 && Math.random() < dropout) continue;
      output.multiplyAdd(input.get(i), weights.get(i));
    }
    if (dropout > 0.0 && dropout < 1.0) {
      output.scale(1.0f / (1.0f - dropout.floatValue()));
    }
    output.multiplyAdd(1.0f, bias.get(outputLayerId));
    applyActivation(output, activationFunction.get(outputLayerId));
    return output;
  }

  private void applyActivation(FloatVector input, FunctionForm func) {
    switch (func) {
      case SIGMOID: {
        input.sigmoid();
        break;
      }
      case RELU: {
        input.rectify();
        break;
      }
      case TANH: {
        input.tanh();
        break;
      }
      case IDENTITY: {
        break;
      }
      default: {
        // set sigmoid activation as default
        input.sigmoid();
      }
    }
  }

  @Override
  public float debugScoreItem(FeatureVector combinedItem,
                              StringBuilder builder) {
    // TODO(peng): implement debug
    return scoreItem(combinedItem);
  }

  @Override
  public List<DebugScoreRecord> debugScoreComponents(FeatureVector combinedItem) {
    // TODO(peng): implement debugScoreComponents
    return new ArrayList<>();
  }

  public void save(BufferedWriter writer) throws IOException {
    ModelHeader header = new ModelHeader();
    header.setModelType("multilayer_perceptron");
    header.setNumHiddenLayers(numHiddenLayers);
    ArrayList<Integer> nodeNum = new ArrayList<>();
    for (int i = 0; i < numHiddenLayers + 1; i++) {
      // this includes the number of node at the output layer
      nodeNum.add(layerNodeNumber.get(i));
    }
    header.setNumberHiddenNodes(nodeNum);
    long count = 0;
    for (Map.Entry<String, Map<String, FloatVector>> familyMap : inputLayerWeights.entrySet()) {
      count += familyMap.getValue().entrySet().size();
    }
    // number of record for the input layer weights
    header.setNumRecords(count);
    ModelRecord headerRec = new ModelRecord();
    headerRec.setModelHeader(header);
    writer.write(Util.encode(headerRec));
    writer.newLine();

    // save the input layer weight, one record per feature
    for (Map.Entry<String, Map<String, FloatVector>> familyMap : inputLayerWeights.entrySet()) {
      for (Map.Entry<String, FloatVector> feature : familyMap.getValue().entrySet()) {
        ModelRecord record = new ModelRecord();
        record.setFeatureFamily(familyMap.getKey());
        record.setFeatureName(feature.getKey());
        ArrayList<Double> arrayList = new ArrayList<>();
        for (int i = 0; i < feature.getValue().length(); i++) {
          arrayList.add((double) feature.getValue().values[i]);
        }
        record.setWeightVector(arrayList);
        writer.write(Util.encode(record));
        writer.newLine();
      }
    }

    // save the bias for each layer after input layer, one record per layer
    for (int i = 0; i < numHiddenLayers + 1; i++) {
      ArrayList<Double> arrayList = new ArrayList<>();
      FloatVector layerBias = bias.get(i);
      int n = layerBias.length();
      ModelRecord record = new ModelRecord();
      for (int j = 0; j < n; j++) {
        arrayList.add((double) layerBias.get(j));
      }
      record.setWeightVector(arrayList);
      record.setFunctionForm(activationFunction.get(i));
      writer.write(Util.encode(record));
      writer.newLine();
    }

    // save the hiddenLayerWeights, one record per (layer + node)
    for (int i = 0; i < numHiddenLayers; i++) {
      ArrayList<FloatVector> weights = hiddenLayerWeights.get(i);
      for (int j = 0; j < layerNodeNumber.get(i); j++) {
        FloatVector w = weights.get(j);
        ModelRecord record = new ModelRecord();
        ArrayList<Double> arrayList = new ArrayList<>();
        for (int k = 0; k < w.length(); k++) {
          arrayList.add((double) w.get(k));
        }
        record.setWeightVector(arrayList);
        writer.write(Util.encode(record));
        writer.newLine();
      }
    }
    writer.flush();
  }

  @Override
  protected void loadInternal(ModelHeader header, BufferedReader reader) throws IOException {
    numHiddenLayers = header.getNumHiddenLayers();
    List<Integer> hiddenNodeNumber = header.getNumberHiddenNodes();
    for (int i = 0; i < hiddenNodeNumber.size(); i++) {
      layerNodeNumber.add(hiddenNodeNumber.get(i));
    }
    // load input layer weights
    long rows = header.getNumRecords();

    for (int i = 0; i < rows; i++) {
      String line = reader.readLine();
      ModelRecord record = Util.decodeModel(line);
      String family = record.getFeatureFamily();
      String name = record.getFeatureName();
      Map<String, FloatVector> inner = inputLayerWeights.get(family);
      if (inner == null) {
        inner = new HashMap<>();
        inputLayerWeights.put(family, inner);
      }
      FloatVector vec = new FloatVector(record.getWeightVector().size());
      for (int j = 0; j < record.getWeightVector().size(); j++) {
        vec.values[j] = record.getWeightVector().get(j).floatValue();
      }
      inner.put(name, vec);
    }
    // load bias and activation function
    for (int i = 0; i < numHiddenLayers + 1; i++) {
      String line = reader.readLine();
      ModelRecord record = Util.decodeModel(line);
      List<Double> arrayList = record.getWeightVector();
      FloatVector layerBias = new FloatVector(arrayList.size());
      for (int j = 0; j < arrayList.size(); j++) {
        layerBias.set(j, arrayList.get(j).floatValue());
      }
      bias.put(i, layerBias);
      activationFunction.add(record.getFunctionForm());
    }

    // load the hiddenLayerWeights, one record per (layer + node)
    for (int i = 0; i < numHiddenLayers; i++) {
      ArrayList<FloatVector> weights = new ArrayList<>();
      for (int j = 0; j < layerNodeNumber.get(i); j++) {
        String line = reader.readLine();
        ModelRecord record = Util.decodeModel(line);
        List<Double> arrayList = record.getWeightVector();
        FloatVector w = new FloatVector(arrayList.size());
        for (int k = 0; k < arrayList.size(); k++) {
          w.set(k, arrayList.get(k).floatValue());
        }
        weights.add(w);
      }
      hiddenLayerWeights.put(i, weights);
    }
  }
}
