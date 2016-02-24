package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.LabelDictionaryEntry;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.MulticlassScoringResult;
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
Test the low rank linear model
*/

public class LowRankLinearModelTest {
  ArrayList<LabelDictionaryEntry> makeLabelDictionary() {
    ArrayList<LabelDictionaryEntry> labelDictionary = new ArrayList<>();
    // construct label dictionary
    LabelDictionaryEntry animalLabel = new LabelDictionaryEntry();
    animalLabel.setLabel("A");
    animalLabel.setCount(1);
    labelDictionary.add(animalLabel);
    LabelDictionaryEntry colorLabel = new LabelDictionaryEntry();
    colorLabel.setLabel("C");
    colorLabel.setCount(1);
    labelDictionary.add(colorLabel);
    LabelDictionaryEntry fruitLabel = new LabelDictionaryEntry();
    fruitLabel.setLabel("F");
    fruitLabel.setCount(1);
    labelDictionary.add(fruitLabel);
    return labelDictionary;
  }

  Map<String, FloatVector> makeLabelWeightVector() {
    Map<String, FloatVector> labelWeights = new HashMap<>();
    float[] animalFeature = {1.0f, 0.0f, 0.0f};
    float[] colorFeature = {0.0f, 1.0f, 0.0f};
    float[] fruitFeature = {0.0f, 0.0f, 1.0f};
    labelWeights.put("A", new FloatVector(animalFeature));
    labelWeights.put("C", new FloatVector(colorFeature));
    labelWeights.put("F", new FloatVector(fruitFeature));
    return labelWeights;
  }

  LowRankLinearModel makeLowRankLinearModel() {
    // A naive model with three classes 'animal', 'color' and 'fruit'
    // and the size of embedding D = number of labels, W is an identity matrix
    LowRankLinearModel model = new LowRankLinearModel();
    model.setEmbeddingDimension(3);
    model.setLabelDictionary(makeLabelDictionary());

    // construct featureWeightVector
    Map<String, Map<String, FloatVector>> featureWeights = new HashMap<>();
    Map<String, FloatVector> animalFeatures = new HashMap<>();
    Map<String, FloatVector> colorFeatures = new HashMap<>();
    Map<String, FloatVector> fruitFeatures = new HashMap<>();

    String[] animalWords = {"cat", "dog", "horse", "fish"};
    String[] colorWords = {"red", "black", "blue", "white", "yellow"};
    String[] fruitWords = {"apple", "kiwi", "pear", "peach"};
    float[] animalFeature = {1.0f, 0.0f, 0.0f};
    float[] colorFeature = {0.0f, 1.0f, 0.0f};
    float[] fruitFeature = {0.0f, 0.0f, 1.0f};

    for (String word: animalWords) {
      animalFeatures.put(word, new FloatVector(animalFeature));
    }

    for (String word: colorWords) {
      colorFeatures.put(word, new FloatVector(colorFeature));
    }

    for (String word: fruitWords) {
      fruitFeatures.put(word, new FloatVector(fruitFeature));
    }
    featureWeights.put("a", animalFeatures);
    featureWeights.put("c", colorFeatures);
    featureWeights.put("f", fruitFeatures);
    model.setFeatureWeightVector(featureWeights);

    // set labelWeightVector
    model.setLabelWeightVector(makeLabelWeightVector());
    model.buildLabelToIndex();

    return model;
  }

  public FeatureVector makeFeatureVector(String label) {
    FeatureVector featureVector = new FeatureVector();
    HashMap stringFeatures = new HashMap<String, HashSet<String>>();
    featureVector.setStringFeatures(stringFeatures);
    HashMap floatFeatures = new HashMap<String, HashMap<String, Double>>();
    featureVector.setFloatFeatures(floatFeatures);
    HashMap feature = new HashMap<String, Float>();
    switch (label) {
      case "animal": {
        feature.put("cat", 1.0);
        feature.put("dog", 2.0);
        floatFeatures.put("a", feature);
        break;
      }
      case "color": {
        feature.put("red", 2.0);
        feature.put("black", 4.0);
        floatFeatures.put("c", feature);
        break;
      }
      case "fruit": {
        feature.put("apple", 1.0);
        feature.put("kiwi", 3.0);
        floatFeatures.put("f", feature);
        break;
      }
      default: break;
    }
    return featureVector;
  }

  @Test
  public void testScoreEmptyFeature() {
    FeatureVector featureVector = new FeatureVector();
    LowRankLinearModel model = makeLowRankLinearModel();
    ArrayList<MulticlassScoringResult> score = model.scoreItemMulticlass(featureVector);
    assertEquals(score.size(), 3);
    assertEquals(0.0f, score.get(0).score, 1e-10f);
    assertEquals(0.0f, score.get(1).score, 1e-10f);
    assertEquals(0.0f, score.get(2).score, 1e-10f);
  }

  @Test
  public void testScoreNonEmptyFeature() {
    FeatureVector animalFv = makeFeatureVector("animal");
    FeatureVector colorFv = makeFeatureVector("color");
    FeatureVector fruitFv = makeFeatureVector("fruit");
    LowRankLinearModel model = makeLowRankLinearModel();

    ArrayList<MulticlassScoringResult> s1 = model.scoreItemMulticlass(animalFv);
    assertEquals(s1.size(), 3);
    assertEquals(0.0f, s1.get(0).score, 3.0f);
    assertEquals(0.0f, s1.get(1).score, 1e-10f);
    assertEquals(0.0f, s1.get(2).score, 1e-10f);

    ArrayList<MulticlassScoringResult> s2 = model.scoreItemMulticlass(colorFv);
    assertEquals(s2.size(), 3);
    assertEquals(0.0f, s2.get(0).score, 1e-10f);
    assertEquals(0.0f, s2.get(1).score, 6.0f);
    assertEquals(0.0f, s2.get(2).score, 1e-10f);

    ArrayList<MulticlassScoringResult> s3 = model.scoreItemMulticlass(fruitFv);
    assertEquals(s3.size(), 3);
    assertEquals(0.0f, s3.get(0).score, 1e-10f);
    assertEquals(0.0f, s3.get(1).score, 1e-10f);
    assertEquals(0.0f, s3.get(2).score, 4.0f);
  }

  @Test
  public void testLoad() {
    CharArrayWriter charWriter = new CharArrayWriter();
    BufferedWriter writer = new BufferedWriter(charWriter);
    ModelHeader header = new ModelHeader();
    header.setModelType("low_rank_linear");
    header.setLabelDictionary(makeLabelDictionary());
    Map<String, FloatVector> labelWeightVector = makeLabelWeightVector();
    Map<String, java.util.List<Double>> labelEmbedding = new HashMap<>();
    for (Map.Entry<String, FloatVector> labelRepresentation : labelWeightVector.entrySet()) {
      float[] values = labelRepresentation.getValue().getValues();

      ArrayList<Double> arrayList = new ArrayList<>();
      for (int i = 0; i < 3; i++) {
        arrayList.add((double) values[i]);
      }
      labelEmbedding.put(labelRepresentation.getKey(), arrayList);
    }
    header.setLabelEmbedding(labelEmbedding);
    header.setNumRecords(4);

    ArrayList<Double> ws = new ArrayList<>();
    ws.add(1.0);
    ws.add(0.0);
    ws.add(0.0);

    ModelRecord record1 = new ModelRecord();
    record1.setModelHeader(header);
    ModelRecord record2 = new ModelRecord();
    record2.setFeatureFamily("a");
    record2.setFeatureName("cat");
    record2.setWeightVector(ws);
    ModelRecord record3 = new ModelRecord();
    record3.setFeatureFamily("a");
    record3.setFeatureName("dog");
    record3.setWeightVector(ws);
    ModelRecord record4 = new ModelRecord();
    record4.setFeatureFamily("a");
    record4.setFeatureName("fish");
    record4.setWeightVector(ws);
    ModelRecord record5 = new ModelRecord();
    record5.setFeatureFamily("a");
    record5.setFeatureName("horse");
    record5.setWeightVector(ws);

    try {
      writer.write(Util.encode(record1) + "\n");
      writer.write(Util.encode(record2) + "\n");
      writer.write(Util.encode(record3) + "\n");
      writer.write(Util.encode(record4) + "\n");
      writer.write(Util.encode(record5) + "\n");
      writer.close();
    } catch (IOException e) {
      assertTrue("Could not write", false);
    }
    String serialized = charWriter.toString();
    assertTrue(serialized.length() > 0);
    StringReader strReader = new StringReader(serialized);
    BufferedReader reader = new BufferedReader(strReader);
    FeatureVector animalFv = makeFeatureVector("animal");
    FeatureVector colorFv = makeFeatureVector("color");
    try {
      Optional<AbstractModel> model = ModelFactory.createFromReader(reader);
      assertTrue(model.isPresent());
      ArrayList<MulticlassScoringResult> s1 = model.get().scoreItemMulticlass(animalFv);
      assertEquals(s1.size(), 3);
      assertEquals(0.0f, s1.get(0).score, 3.0f);
      assertEquals(0.0f, s1.get(1).score, 1e-10f);
      assertEquals(0.0f, s1.get(2).score, 1e-10f);
      ArrayList<MulticlassScoringResult> s2 = model.get().scoreItemMulticlass(colorFv);
      assertEquals(s2.size(), 3);
      assertEquals(0.0f, s2.get(0).score, 1e-10f);
      assertEquals(0.0f, s2.get(1).score, 1e-10f);
      assertEquals(0.0f, s2.get(2).score, 1e-10f);
    } catch (IOException e) {
      assertTrue("Could not read", false);
    }
  }

  @Test
  public void testSave() {
    StringWriter strWriter = new StringWriter();
    BufferedWriter writer = new BufferedWriter(strWriter);
    LowRankLinearModel model = makeLowRankLinearModel();
    try {
      model.save(writer);
      writer.close();
    } catch (IOException e) {
      assertTrue("Could not save", false);
    }
  }
}

