package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.util.Util;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;
import java.util.HashMap;
import com.google.common.hash.HashCode;
import com.google.common.base.Optional;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class LinearModelTest {
  private static final Logger log = LoggerFactory.getLogger(LinearModelTest.class);

  public FeatureVector makeFeatureVector() {
    Set list = new HashSet<String>();
    list.add("aaa");
    list.add("bbb");
    // add a feature that is missing in the model
    list.add("ccc");
    HashMap stringFeatures = new HashMap<String, ArrayList<String>>();
    stringFeatures.put("string_feature", list);
    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    return featureVector;
  }

  public LinearModel makeLinearModel() {
    LinearModel model = new LinearModel();
    Map<String, Map<String, Float>> weights = new HashMap<>();
    Map<String, Float> inner = new HashMap<>();
    weights.put("string_feature", inner);
    inner.put("aaa", 0.5f);
    inner.put("bbb", 0.25f);
    model.setWeights(weights);
    model.setOffset(0.5f);
    model.setSlope(1.5f);
    return model;
  }

  @Test
  public void testScoreEmptyFeature() {
    FeatureVector featureVector = new FeatureVector();
    LinearModel model = new LinearModel();
    float score = model.scoreItem(featureVector);
    assertTrue(score < 1e-10f);
    assertTrue(score > -1e-10f);
  }

  @Test
  public void testScoreNonEmptyFeature() {
    FeatureVector featureVector = makeFeatureVector();
    LinearModel model = new LinearModel();
    Map<String, Map<String, Float>> weights = new HashMap<>();
    Map<String, Float> inner = new HashMap<>();
    weights.put("string_feature", inner);
    inner.put("aaa", 0.5f);
    inner.put("bbb", 0.25f);
    model.setWeights(weights);
    float score = model.scoreItem(featureVector);
    assertTrue(score < 0.76f);
    assertTrue(score > 0.74f);
  }

  @Test
  public void testLoad() {
    CharArrayWriter charWriter = new CharArrayWriter();
    BufferedWriter writer = new BufferedWriter(charWriter);
    ModelHeader header = new ModelHeader();
    header.setModelType("linear");
    header.setNumRecords(1);
    ModelRecord record1 = new ModelRecord();
    record1.setModelHeader(header);
    ModelRecord record2 = new ModelRecord();
    record2.setFeatureFamily("string_feature");
    record2.setFeatureName("bbb");
    record2.setFeatureWeight(0.9f);
    try {
      writer.write(Util.encode(record1) + "\n");
      writer.write(Util.encode(record2) + "\n");
      writer.close();
    } catch (IOException e) {
       assertTrue("Could not write", false);
    }
    String serialized = charWriter.toString();
    assertTrue(serialized.length() > 0);
    StringReader strReader = new StringReader(serialized);
    BufferedReader reader = new BufferedReader(strReader);
    FeatureVector featureVector = makeFeatureVector();
    try {
      Optional<AbstractModel> model = ModelFactory.createFromReader(reader);
      assertTrue(model.isPresent());
      float score = model.get().scoreItem(featureVector);
      assertTrue(score > 0.89f);
      assertTrue(score < 0.91f);
    } catch (IOException e) {
      assertTrue("Could not read", false);
    }
  }

  @Test
  public void testSave() {
    StringWriter strWriter = new StringWriter();
    BufferedWriter writer = new BufferedWriter(strWriter);
    LinearModel model = makeLinearModel();
    try {
      model.save(writer);
      writer.close();
    } catch (IOException e) {
      assertTrue("Could not save", false);
    }
  }

  @Test
  public void testDebugScoreComponents() {
    LinearModel model = makeLinearModel();
    FeatureVector fv = makeFeatureVector();
    List<DebugScoreRecord> scoreRecordsList = model.debugScoreComponents(fv);
    assertTrue(scoreRecordsList.size() == 2);
    for (DebugScoreRecord record : scoreRecordsList) {
      assertTrue(record.featureFamily == "string_feature");
      assertTrue(record.featureName == "aaa" || record.featureName == "bbb");
      assertTrue(record.featureValue == 1.0);
      if (record.featureName == "aaa") {
        assertTrue(record.featureWeight == 0.5f);
      } else {
        assertTrue(record.featureWeight == 0.25f);
      }
    }
  }
}