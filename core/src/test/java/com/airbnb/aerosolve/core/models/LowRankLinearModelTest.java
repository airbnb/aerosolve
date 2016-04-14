package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.LabelDictionaryEntry;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.MulticlassScoringResult;
import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.TransformTestingHelper;
import com.airbnb.aerosolve.core.util.FloatVector;
import com.airbnb.aerosolve.core.util.Util;
import com.google.common.base.Optional;
import it.unimi.dsi.fastutil.objects.Reference2ObjectMap;
import it.unimi.dsi.fastutil.objects.Reference2ObjectOpenHashMap;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.CharArrayWriter;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/*
Test the low rank linear model
*/

public class LowRankLinearModelTest {
  private final FeatureRegistry registry = new FeatureRegistry();

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
    LowRankLinearModel model = new LowRankLinearModel(registry);
    model.embeddingDimension(3);
    model.labelDictionary(makeLabelDictionary());

    // construct featureWeightVector
    Reference2ObjectMap<Feature, FloatVector> featureWeights = new Reference2ObjectOpenHashMap<>();
    Map<String, FloatVector> animalFeatures = new HashMap<>();
    Map<String, FloatVector> colorFeatures = new HashMap<>();
    Map<String, FloatVector> fruitFeatures = new HashMap<>();

    String[] animalWords = {"cat", "dog", "horse", "fish"};
    String[] colorWords = {"red", "black", "blue", "white", "yellow"};
    String[] fruitWords = {"apple", "kiwi", "pear", "peach"};
    float[] animalFeature = {1.0f, 0.0f, 0.0f};
    float[] colorFeature = {0.0f, 1.0f, 0.0f};
    float[] fruitFeature = {0.0f, 0.0f, 1.0f};

    Family animalFamily = registry.family("a");
    for (String word: animalWords) {
      featureWeights.put(animalFamily.feature(word), new FloatVector(animalFeature));
    }

    Family colorFamily = registry.family("c");
    for (String word: colorWords) {
      featureWeights.put(colorFamily.feature(word), new FloatVector(colorFeature));
    }

    Family fruitFamily = registry.family("f");
    for (String word: fruitWords) {
      featureWeights.put(fruitFamily.feature(word), new FloatVector(fruitFeature));
    }
    model.featureWeightVector(featureWeights);

    // set labelWeightVector
    model.labelWeightVector(makeLabelWeightVector());
    model.buildLabelToIndex();

    return model;
  }

  public FeatureVector makeFeatureVector(String label) {
    MultiFamilyVector featureVector = TransformTestingHelper.makeEmptyVector(registry);
    switch (label) {
      case "animal": {
        Family family = registry.family("a");
        featureVector.put(family.feature("cat"), 1.0);
        featureVector.put(family.feature("dog"), 2.0);
        break;
      }
      case "color": {
        Family family = registry.family("c");
        featureVector.put(family.feature("red"), 2.0);
        featureVector.put(family.feature("black"), 4.0);
        break;
      }
      case "fruit": {
        Family family = registry.family("f");
        featureVector.put(family.feature("apple"), 1.0);
        featureVector.put(family.feature("kiwi"), 3.0);
        break;
      }
      default: break;
    }

    return featureVector;
  }

  @Test
  public void testScoreEmptyFeature() {
    MultiFamilyVector featureVector = TransformTestingHelper.makeEmptyVector(registry);
    LowRankLinearModel model = makeLowRankLinearModel();
    ArrayList<MulticlassScoringResult> score = model.scoreItemMulticlass(featureVector);
    assertEquals(score.size(), 3);
    assertEquals(0.0d, score.get(0).getScore(), 1e-10d);
    assertEquals(0.0d, score.get(1).getScore(), 1e-10d);
    assertEquals(0.0d, score.get(2).getScore(), 1e-10d);
  }

  @Test
  public void testScoreNonEmptyFeature() {
    FeatureVector animalFv = makeFeatureVector("animal");
    FeatureVector colorFv = makeFeatureVector("color");
    FeatureVector fruitFv = makeFeatureVector("fruit");
    LowRankLinearModel model = makeLowRankLinearModel();

    ArrayList<MulticlassScoringResult> s1 = model.scoreItemMulticlass(animalFv);
    assertEquals(s1.size(), 3);
    assertEquals(3.0d, s1.get(0).getScore(), 1e-10d);
    assertEquals(0.0d, s1.get(1).getScore(), 1e-10d);
    assertEquals(0.0d, s1.get(2).getScore(), 1e-10d);

    ArrayList<MulticlassScoringResult> s2 = model.scoreItemMulticlass(colorFv);
    assertEquals(s2.size(), 3);
    assertEquals(0.0d, s2.get(0).getScore(), 1e-10d);
    assertEquals(6.0d, s2.get(1).getScore(), 1e-10d);
    assertEquals(0.0d, s2.get(2).getScore(), 1e-10d);

    ArrayList<MulticlassScoringResult> s3 = model.scoreItemMulticlass(fruitFv);
    assertEquals(s3.size(), 3);
    assertEquals(0.0d, s3.get(0).getScore(), 1e-10d);
    assertEquals(0.0d, s3.get(1).getScore(), 1e-10d);
    assertEquals(4.0d, s3.get(2).getScore(), 1e-10d);
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
      Optional<AbstractModel> model = ModelFactory.createFromReader(reader, registry);
      assertTrue(model.isPresent());
      ArrayList<MulticlassScoringResult> s1 = model.get().scoreItemMulticlass(animalFv);
      assertEquals(s1.size(), 3);
      assertEquals(3.0d, s1.get(0).getScore(), 1e-10d);
      assertEquals(0.0d, s1.get(1).getScore(), 1e-10d);
      assertEquals(0.0d, s1.get(2).getScore(), 1e-10d);
      ArrayList<MulticlassScoringResult> s2 = model.get().scoreItemMulticlass(colorFv);
      assertEquals(s2.size(), 3);
      assertEquals(0.0d, s2.get(0).getScore(), 1e-10d);
      assertEquals(0.0d, s2.get(1).getScore(), 1e-10d);
      assertEquals(0.0d, s2.get(2).getScore(), 1e-10d);
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

