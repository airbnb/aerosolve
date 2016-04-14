package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.TransformTestingHelper;
import com.airbnb.aerosolve.core.util.Util;
import com.google.common.base.Optional;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.CharArrayWriter;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
@Slf4j
public class LinearModelTest {
  private final FeatureRegistry registry = new FeatureRegistry();

  public MultiFamilyVector makeFeatureVector() {
   return TransformTestingHelper.builder(registry)
       .string("string_feature", "aaa")
       .string("string_feature", "bbb")
       .string("string_feature", "ccc")
        .build();
  }

  public LinearModel makeLinearModel() {
    LinearModel model = new LinearModel(registry);
    Family stringFamily = registry.family("string_feature");
    model.weights().put(stringFamily.feature("aaa"), 0.5d);
    model.weights().put(stringFamily.feature("bbb"), 0.25d);
    model.offset(0.5d);
    model.slope(1.5d);
    return model;
  }

  @Test
  public void testScoreEmptyFeature() {
    MultiFamilyVector featureVector = TransformTestingHelper.makeEmptyVector(registry);
    LinearModel model = new LinearModel(registry);
    double score = model.scoreItem(featureVector);
    assertTrue(score < 1e-10f);
    assertTrue(score > -1e-10f);
  }

  @Test
  public void testScoreNonEmptyFeature() {
    FeatureVector featureVector = makeFeatureVector();
    LinearModel model = makeLinearModel();
    double score = model.scoreItem(featureVector);
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
      Optional<AbstractModel> model = ModelFactory.createFromReader(reader, registry);
      assertTrue(model.isPresent());
      double score = model.get().scoreItem(featureVector);
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
      assertTrue("string_feature".equals(record.getFeatureFamily()));
      assertTrue("aaa".equals(record.getFeatureName()) || "bbb".equals(record.getFeatureName()));
      assertTrue(record.getFeatureValue() == 1.0);
      if ("aaa".equals(record.getFeatureName())) {
        assertTrue(record.getFeatureWeight() == 0.5d);
      } else {
        assertTrue(record.getFeatureWeight() == 0.25d);
      }
    }
  }
}