package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.function.Function;
import com.airbnb.aerosolve.core.function.Linear;
import com.airbnb.aerosolve.core.function.Spline;
import com.airbnb.aerosolve.core.util.Util;
import com.google.common.base.Optional;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertEquals;
/**
 * Test the additive model
 */
public class AdditiveModelTest {
  private static final Logger log = LoggerFactory.getLogger(AdditiveModelTest.class);
  AdditiveModel makeAdditiveModel() {
    AdditiveModel model = new AdditiveModel();
    Map<String, Map<String, Function>> weights = new HashMap<>();
    Map<String, Function> innerSplineFloat = new HashMap<String, Function>();
    Map<String, Function> innerLinearFloat = new HashMap<String, Function>();
    Map<String, Function> innerSplineString = new HashMap<String, Function>();
    Map<String, Function> innerLinearString = new HashMap<String, Function>();
    weights.put("spline_float", innerSplineFloat);
    weights.put("linear_float", innerLinearFloat);
    weights.put("spline_string", innerSplineString);
    weights.put("linear_string", innerLinearString);
    float [] ws = {5.0f, 10.0f, -20.0f};
    innerSplineFloat.put("aaa", new Spline(1.0f, 3.0f, ws));
    // for string feature, only the first element in weight is meaningful.
    innerSplineString.put("bbb", new Spline(1.0f, 2.0f, ws));
    float [] wl = {1.0f, 2.0f};
    innerLinearFloat.put("ccc", new Linear(-10.0f, 5.0f, wl));
    innerLinearString.put("ddd", new Linear(1.0f, 1.0f, wl));
    model.setWeights(weights);
    model.setOffset(0.5f);
    model.setSlope(1.5f);
    return model;
  }

  public FeatureVector makeFeatureVector(float a, float c) {
    FeatureVector featureVector = new FeatureVector();
    HashMap stringFeatures = new HashMap<String, HashSet<String>>();
    featureVector.setStringFeatures(stringFeatures);
    HashMap floatFeatures = new HashMap<String, HashMap<String, Double>>();
    featureVector.setFloatFeatures(floatFeatures);
    // prepare string features
    Set list1 = new HashSet<String>();
    list1.add("bbb"); // weight = 5.0f
    list1.add("ggg"); // this feature is missing in the model
    stringFeatures.put("spline_string", list1);

    Set list2 = new HashSet<String>();
    list2.add("ddd"); // weight = 3.0f
    list2.add("ggg"); // this feature is missing in the model
    stringFeatures.put("linear_string", list2);
    featureVector.setStringFeatures(stringFeatures);
    // prepare float features
    HashMap splineFloat = new HashMap<String, Double>();
    HashMap linearFloat = new HashMap<String, Double>();
    floatFeatures.put("spline_float", splineFloat);
    floatFeatures.put("linear_float", linearFloat);

    splineFloat.put("aaa", (double) a); // corresponds to Spline(1.0f, 3.0f, {5, 10, -20})
    splineFloat.put("ggg", 1.0); // missing features
    linearFloat.put("ccc", (double) c);   // weight = 1+2*c
    linearFloat.put("ggg", 10.0); // missing features

    return featureVector;
  }

  @Test
  public void testScoreEmptyFeature() {
    FeatureVector featureVector = new FeatureVector();
    AdditiveModel model = new AdditiveModel();
    float score = model.scoreItem(featureVector);
    assertEquals(0.0f, score, 1e-10f);
  }

  @Test
  public void testScoreNonEmptyFeature() {
    AdditiveModel model = makeAdditiveModel();

    FeatureVector fv1 = makeFeatureVector(1.0f, 0.0f);
    float score1 = model.scoreItem(fv1);
    assertEquals(8.0f + 5.0f + (1.0f + 2.0f * (0.0f + 10.0f) / 15.0f), score1, 0.001f);

    FeatureVector fv2 = makeFeatureVector(-1.0f, 0.0f);
    float score2 = model.scoreItem(fv2);
    assertEquals(8.0f + 5.0f + (1.0f + 2.0f * (0.0f + 10.0f) / 15.0f), score2, 0.001f);

    FeatureVector fv3 = makeFeatureVector(4.0f, 1.0f);
    float score3 = model.scoreItem(fv3);
    assertEquals(8.0f - 20.0f + (1.0f + 2.0f * (1.0f + 10.0f) / 15.0f), score3, 0.001f);

    FeatureVector fv4 = makeFeatureVector(2.0f, 7.0f);
    float score4 = model.scoreItem(fv4);
    assertEquals(8.0f + 10.0f + (1.0f + 2.0f * (7.0f + 10.0f) / 15.0f), score4, 0.001f);
  }

  @Test
  public void testLoad() {
    CharArrayWriter charWriter = new CharArrayWriter();
    BufferedWriter writer = new BufferedWriter(charWriter);
    ModelHeader header = new ModelHeader();
    header.setModelType("additive");
    header.setNumRecords(4);

    ArrayList<Double> ws = new ArrayList<Double>();
    ws.add(5.0);
    ws.add(10.0);
    ws.add(-20.0);
    ArrayList<Double> wl = new ArrayList<Double>();
    wl.add(1.0);
    wl.add(2.0);

    ModelRecord record1 = new ModelRecord();
    record1.setModelHeader(header);
    ModelRecord record2 = new ModelRecord();
    record2.setFunctionForm(FunctionForm.Spline);
    record2.setFeatureFamily("spline_float");
    record2.setFeatureName("aaa");
    record2.setWeightVector(ws);
    record2.setMinVal(1.0);
    record2.setMaxVal(3.0);
    ModelRecord record3 = new ModelRecord();
    record3.setFunctionForm(FunctionForm.Spline);
    record3.setFeatureFamily("spline_string");
    record3.setFeatureName("bbb");
    record3.setWeightVector(ws);
    record3.setMinVal(1.0);
    record3.setMaxVal(2.0);
    ModelRecord record4 = new ModelRecord();
    record4.setFunctionForm(FunctionForm.Linear);
    record4.setFeatureFamily("linear_float");
    record4.setFeatureName("ccc");
    record4.setWeightVector(wl);
    ModelRecord record5 = new ModelRecord();
    record5.setFunctionForm(FunctionForm.Linear);
    record5.setFeatureFamily("linear_string");
    record5.setFeatureName("ddd");
    record5.setWeightVector(wl);

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
    FeatureVector featureVector = makeFeatureVector(2.0f, 7.0f);
    try {
      Optional<AbstractModel> model = ModelFactory.createFromReader(reader);
      assertTrue(model.isPresent());
      float score = model.get().scoreItem(featureVector);
      assertEquals(8.0f + 10.0f + 15.0f, score, 0.001f);
    } catch (IOException e) {
      assertTrue("Could not read", false);
    }
  }

  @Test
  public void testSave() {
    StringWriter strWriter = new StringWriter();
    BufferedWriter writer = new BufferedWriter(strWriter);
    AdditiveModel model = makeAdditiveModel();
    try {
      model.save(writer);
      writer.close();
    } catch (IOException e) {
      assertTrue("Could not save", false);
    }
  }

  @Test
  public void testAddFunction() {
    AdditiveModel model = makeAdditiveModel();
    // add an existing feature without overwrite
    model.addFunction("spline_float", "aaa", new Spline(2.0f, 10.0f, 5), false);
    // add an existing feature with overwrite
    model.addFunction("linear_float", "ccc", new Linear(3.0f, 5.0f), true);
    // add a new feature
    model.addFunction("spline_float", "new", new Spline(2.0f, 10.0f, 5), false);

    Map<String, Map<String, Function>> weights = model.getWeights();
    for (Map.Entry<String, Map<String, Function>>  featureFamily: weights.entrySet()) {
      String familyName = featureFamily.getKey();
      Map<String, Function> features = featureFamily.getValue();
      for (Map.Entry<String, Function> feature: features.entrySet()) {
        String featureName = feature.getKey();
        Function func = feature.getValue();
        if (familyName.equals("spline_float")) {
          Spline spline = (Spline) func;
          if (featureName.equals("aaa")) {
            assertTrue(spline.getMaxVal() == 3.0f);
            assertTrue(spline.getMinVal() == 1.0f);
            assertTrue(spline.getWeights().length == 3);
          } else if (featureName.equals("new")) {
            assertTrue(spline.getMaxVal() == 10.0f);
            assertTrue(spline.getMinVal() == 2.0f);
            assertTrue(spline.getWeights().length == 5);
          }
         } else if(familyName.equals("linear_float") && featureName.equals("ccc")) {
          Linear linear = (Linear) func;
          assertTrue(linear.getWeights().length == 2);
          assertTrue(linear.getWeights()[0] == 0.0f);
          assertTrue(linear.getWeights()[1] == 0.0f);
          assertTrue(linear.getMinVal() == 3.0f);
          assertTrue(linear.getMaxVal() == 5.0f);
        }
      }
    }
  }

  @Test
  public void testModelClone() throws CloneNotSupportedException {
    float[] weights = {5.0f, 10.0f, -20.0f};
    AdditiveModel model = makeAdditiveModel();
    model.addFunction("spline_float", "aaa", new Spline(2.0f, 10.0f, weights), false);

    AdditiveModel modelClone = model.clone();
    modelClone.getWeights().get("spline_float").get("aaa").resample(2);

    assertArrayEquals(((Spline)model.getWeights().get("spline_float").get("aaa")).getWeights(), weights, 0);
  }
}
