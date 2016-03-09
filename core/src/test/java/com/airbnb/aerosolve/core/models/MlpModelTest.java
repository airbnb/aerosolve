package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.util.FloatVector;
import com.airbnb.aerosolve.core.util.Util;
import com.google.common.base.Optional;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.CharArrayWriter;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertEquals;

/*
Test the MLP model
*/
public class MlpModelTest {
  public FeatureVector makeFeatureVector() {
    FeatureVector featureVector = new FeatureVector();
    HashMap stringFeatures = new HashMap<String, HashSet<String>>();
    featureVector.setStringFeatures(stringFeatures);
    HashMap floatFeatures = new HashMap<String, HashMap<String, Double>>();
    featureVector.setFloatFeatures(floatFeatures);
    HashMap feature = new HashMap<String, Float>();
    feature.put("a", 1.0);
    feature.put("b", 2.0);
    floatFeatures.put("in", feature);
    return featureVector;
  }
  public MlpModel makeMlpModel(FunctionForm func) {
    // construct a network with 1 hidden layer
    // and there are 3 nodes in the hidden layer
    ArrayList nodeNum = new ArrayList(2);
    nodeNum.add(3);
    nodeNum.add(1);
    // assume bias at each node are zeros
    ArrayList activations = new ArrayList();
    activations.add(func);
    activations.add(func);
    MlpModel model = new MlpModel(activations, nodeNum);

    // set input layer
    HashMap inputLayer = new HashMap<>();
    HashMap inner = new HashMap<>();
    FloatVector f11 = new FloatVector(3);
    f11.set(0, 0.0f);
    f11.set(1, 1.0f);
    f11.set(2, 1.0f);
    FloatVector f12 = new FloatVector(3);
    f12.set(0, 1.0f);
    f12.set(1, 1.0f);
    f12.set(2, 0.0f);
    inner.put("a", f11);
    inner.put("b", f12);
    inputLayer.put("in", inner);
    model.setInputLayerWeights(inputLayer);
    // set hidden layer
    HashMap hiddenLayer = new HashMap<>();
    FloatVector f21 = new FloatVector(1);
    FloatVector f22 = new FloatVector(1);
    FloatVector f23 = new FloatVector(1);
    f21.set(0, 0.5f);
    f22.set(0, 1.0f);
    f23.set(0, 2.0f);
    ArrayList hidden = new ArrayList(3);
    hidden.add(f21);
    hidden.add(f22);
    hidden.add(f23);
    hiddenLayer.put(0, hidden);
    model.setHiddenLayerWeights(hiddenLayer);
    return model;
  }

  @Test
  public void testConstructedModel() {
    MlpModel model = makeMlpModel(FunctionForm.RELU);
    assertEquals(model.getNumHiddenLayers(), 1);
    assertEquals(model.getActivationFunction().get(0), FunctionForm.RELU);
    assertEquals(model.getHiddenLayerWeights().size(), 1);
    assertEquals(model.getHiddenLayerWeights().get(0).size(), 3);
    assertEquals(model.getInputLayerWeights().entrySet().size(), 1);
    assertEquals(model.getInputLayerWeights().get("in").entrySet().size(), 2);
    assertEquals(model.getInputLayerWeights().get("in").get("a").length(), 3);
    assertEquals(model.getInputLayerWeights().get("in").get("b").length(), 3);
  }

  @Test
  public void testScoring() {
    FeatureVector fv = makeFeatureVector();
    MlpModel model = makeMlpModel(FunctionForm.RELU);
    float output = model.scoreItem(fv);
    assertEquals(output, 6.0f, 1e-10f);
  }

  @Test
  public void testSave() {
    StringWriter strWriter = new StringWriter();
    BufferedWriter writer = new BufferedWriter(strWriter);
    MlpModel model = makeMlpModel(FunctionForm.RELU);
    try {
      model.save(writer);
      writer.close();
    } catch (IOException e) {
      assertTrue("Could not save", false);
    }
  }

  @Test
  public void testLoad() {
    CharArrayWriter charWriter = new CharArrayWriter();
    BufferedWriter writer = new BufferedWriter(charWriter);
    // create header record
    ModelHeader header = new ModelHeader();
    header.setModelType("multilayer_perceptron");
    header.setNumHiddenLayers(1);
    ArrayList<Integer> nodeNum = new ArrayList<>(2);
    nodeNum.add(3);
    nodeNum.add(1);
    header.setNumberHiddenNodes(nodeNum);
    header.setNumRecords(2);
    ModelRecord record1 = new ModelRecord();
    record1.setModelHeader(header);

    // create records for input layer
    ModelRecord record2 = new ModelRecord();
    record2.setFeatureFamily("in");
    record2.setFeatureName("a");
    ArrayList<Double> in1 = new ArrayList<>();
    in1.add(0.0);
    in1.add(1.0);
    in1.add(1.0);
    record2.setWeightVector(in1);

    ModelRecord record3 = new ModelRecord();
    record3.setFeatureFamily("in");
    record3.setFeatureName("b");
    ArrayList<Double> in2 = new ArrayList<>();
    in2.add(1.0);
    in2.add(1.0);
    in2.add(0.0);
    record3.setWeightVector(in2);

    // create records for bias
    ModelRecord record4 = new ModelRecord();
    ArrayList<Double> b1 = new ArrayList<>();
    b1.add(0.0);
    b1.add(0.0);
    b1.add(0.0);
    record4.setWeightVector(b1);
    record4.setFunctionForm(FunctionForm.RELU);
    ModelRecord record5 = new ModelRecord();
    ArrayList<Double> b2 = new ArrayList<>();
    b2.add(0.0);
    record5.setWeightVector(b2);
    record5.setFunctionForm(FunctionForm.RELU);

    // create records for hidden layer
    ModelRecord record6 = new ModelRecord();
    ArrayList<Double> h1 = new ArrayList<>();
    h1.add(0.5);
    record6.setWeightVector(h1);
    ModelRecord record7 = new ModelRecord();
    ArrayList<Double> h2 = new ArrayList<>();
    h2.add(1.0);
    record7.setWeightVector(h2);
    ModelRecord record8 = new ModelRecord();
    ArrayList<Double> h3 = new ArrayList<>();
    h3.add(2.0);
    record8.setWeightVector(h3);

    try {
      writer.write(Util.encode(record1) + "\n");
      writer.write(Util.encode(record2) + "\n");
      writer.write(Util.encode(record3) + "\n");
      writer.write(Util.encode(record4) + "\n");
      writer.write(Util.encode(record5) + "\n");
      writer.write(Util.encode(record6) + "\n");
      writer.write(Util.encode(record7) + "\n");
      writer.write(Util.encode(record8) + "\n");
      writer.close();
    } catch (IOException e) {
      assertTrue("Could not write", false);
    }
    String serialized = charWriter.toString();
    assertTrue(serialized.length() > 0);
    StringReader strReader = new StringReader(serialized);
    BufferedReader reader = new BufferedReader(strReader);
    FeatureVector fv = makeFeatureVector();
    try {
      Optional<AbstractModel> model = ModelFactory.createFromReader(reader);
      assertTrue(model.isPresent());
      float s = model.get().scoreItem(fv);
      assertEquals(s, 6.0f, 1e-10f);
    } catch (IOException e) {
      assertTrue("Could not read", false);
    }
  }
}
