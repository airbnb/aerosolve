package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.transforms.LegacyNames;
import com.airbnb.aerosolve.core.util.FloatVector;
import com.airbnb.aerosolve.core.util.Util;
import it.unimi.dsi.fastutil.objects.Reference2ObjectMap;
import it.unimi.dsi.fastutil.objects.Reference2ObjectOpenHashMap;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;


/**
 * Multilayer perceptron (MLP) model https://en.wikipedia.org/wiki/Multilayer_perceptron
 * The current implementation is for the case where there is only one output node.
 */
@LegacyNames("multilayer_perceptron")
@Accessors(fluent = true, chain = true)
public class MlpModel extends AbstractModel {

  private static final long serialVersionUID = -6870862764598907090L;

  // weights that define the projection from input layer (layer0) to the first hidden layer
  // or the output layer
  @Getter
  @Setter
  private Reference2ObjectMap<Feature, FloatVector> inputLayerWeights;

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

  public MlpModel(FeatureRegistry registry) {
    super(registry);
    layerNodeNumber = new ArrayList<>();
    inputLayerWeights = new Reference2ObjectOpenHashMap<>();
    hiddenLayerWeights = new HashMap<>();
    layerActivations = new HashMap<>();
    bias = new HashMap<>();
    activationFunction = new ArrayList<>();
  }

  public MlpModel(ArrayList<FunctionForm> activation, ArrayList<Integer> nodeNumbers,
                  FeatureRegistry registry) {
    super(registry);
    // n is the number of hidden layers (including output layer, excluding input layer)
    // activation specifies activation function
    // nodeNumbers: specifies number of nodes in each hidden layer
    numHiddenLayers = nodeNumbers.size() - 1; // excluding output layer
    activationFunction = activation;
    layerNodeNumber = nodeNumbers;
    assert(activation.size() == numHiddenLayers + 1);
    inputLayerWeights = new Reference2ObjectOpenHashMap<>();
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
  public double scoreItem(FeatureVector combinedItem) {
    return forwardPropagation(combinedItem);
  }

  public double forwardPropagation(FeatureVector vector) {
    projectInputLayer(vector, 0.0);
    for (int i = 0; i < numHiddenLayers; i++) {
      projectHiddenLayer(i, 0.0);
    }
    return layerActivations.get(numHiddenLayers).get(0);
  }

  public double forwardPropagationWithDropout(FeatureVector vector, Double dropout) {
    // reference: George E. Dahl et al. "IMPROVING DEEP NEURAL NETWORKS FOR LVCSR USING RECTIFIED LINEAR UNITS AND DROPOUT"
    // scale the input to a node by 1/(1-dropout), so that we don't need to rescale model weights after training
    // make sure the value is between 0 and 1
    assert(dropout > 0.0);
    assert(dropout < 1.0);
    projectInputLayer(vector, dropout);
    for (int i = 0; i < numHiddenLayers; i++) {
      projectHiddenLayer(i, dropout);
    }
    return layerActivations.get(numHiddenLayers).get(0);
  }

  public FloatVector projectInputLayer(FeatureVector vector, Double dropout) {
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

    for (FeatureValue value : vector) {
      FloatVector vec = inputLayerWeights.get(value.feature());
      if (vec != null) {
        if (dropout > 0.0 && Math.random() < dropout) continue;
        fvProjection.multiplyAdd(value.value(), vec);
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
  public double debugScoreItem(FeatureVector combinedItem,
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
    // number of record for the input layer weights
    header.setNumRecords(inputLayerWeights.size());
    ModelRecord headerRec = new ModelRecord();
    headerRec.setModelHeader(header);
    writer.write(Util.encode(headerRec));
    writer.newLine();

    // save the input layer weight, one record per feature
    for (Map.Entry<Feature, FloatVector> entry : inputLayerWeights.entrySet()) {
      ModelRecord record = new ModelRecord();
      Feature feature = entry.getKey();
      record.setFeatureFamily(feature.family().name());
      record.setFeatureName(feature.name());
      ArrayList<Double> arrayList = new ArrayList<>();
      for (int i = 0; i < entry.getValue().length(); i++) {
        arrayList.add((double) entry.getValue().values[i]);
      }
      record.setWeightVector(arrayList);
      writer.write(Util.encode(record));
      writer.newLine();
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
      FloatVector vec = new FloatVector(record.getWeightVector().size());
      for (int j = 0; j < record.getWeightVector().size(); j++) {
        vec.values[j] = record.getWeightVector().get(j).floatValue();
      }
      Feature feature = registry.feature(family, name);
      inputLayerWeights.put(feature, vec);
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
