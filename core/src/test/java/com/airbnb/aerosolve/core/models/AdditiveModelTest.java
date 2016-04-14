package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.functions.Function;
import com.airbnb.aerosolve.core.functions.Linear;
import com.airbnb.aerosolve.core.functions.Spline;
import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.transforms.TransformTestingHelper;
import com.airbnb.aerosolve.core.util.Util;
import com.google.common.base.Optional;
import it.unimi.dsi.fastutil.objects.Reference2ObjectMap;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.CharArrayWriter;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
/**
 * Test the additive model
 */
@Slf4j
public class AdditiveModelTest {
  private final FeatureRegistry registry = new FeatureRegistry();
  AdditiveModel makeAdditiveModel() {
    AdditiveModel model = new AdditiveModel(registry);
    Map<Feature, Function> weights = model.weights();
    double [] ws = {5.0d, 10.0d, -20.0d};
    weights.put(registry.feature("spline_float", "aaa"), new Spline(1.0d, 3.0d, ws));

    // for string feature, only the first element in weight is meaningful.
    weights.put(registry.feature("spline_string", "bbb"), new Spline(1.0d, 2.0d, ws));
    double [] wl = {1.0d, 2.0d};
    weights.put(registry.feature("linear_float", "ccc"), new Linear(-10.0d, 5.0d, wl));
    weights.put(registry.feature("linear_string", "ddd"), new Linear(1.0d, 1.0d, wl));
    model.offset(0.5f);
    model.slope(1.5f);
    return model;
  }

  public FeatureVector makeFeatureVector(double a, double c) {
    FeatureVector featureVector = TransformTestingHelper.makeEmptyVector(registry);

    // prepare string features
    Family splineStringFamily = registry.family("spline_string");
    featureVector.putString(splineStringFamily.feature("bbb")); // weight = 5.0d
    featureVector.putString(splineStringFamily.feature("ggg")); // this feature is missing in the model

    Family linearStringFamily = registry.family("linear_string");
    featureVector.putString(linearStringFamily.feature("ddd")); // weight = 3.0d
    featureVector.putString(linearStringFamily.feature("ggg")); // this feature is missing in the model

    // prepare float features
    Family splineFloat = registry.family("spline_float");
    featureVector.put(splineFloat.feature("aaa"), a); // corresponds to Spline(1.0d, 3.0d, {5, 10, -20})
    featureVector.put(splineFloat.feature("ggg"), 1.0); // missing features

    Family linearFloat = registry.family("linear_float");
    featureVector.put(linearFloat.feature("ccc"), c); // weight = 1+2*c
    featureVector.put(splineFloat.feature("ggg"), 10.0); // missing features

    return featureVector;
  }

  @Test
  public void testScoreEmptyFeature() {
    FeatureVector featureVector = TransformTestingHelper.makeEmptyVector(registry);
    AdditiveModel model = new AdditiveModel(registry);
    double score = model.scoreItem(featureVector);
    assertEquals(0.0d, score, 1e-10d);
  }

  @Test
  public void testScoreNonEmptyFeature() {
    AdditiveModel model = makeAdditiveModel();

    FeatureVector fv1 = makeFeatureVector(1.0d, 0.0d);
    double score1 = model.scoreItem(fv1);
    assertEquals(8.0d + 5.0d + (1.0d + 2.0d * (0.0d + 10.0d) / 15.0d), score1, 0.001d);

    FeatureVector fv2 = makeFeatureVector(-1.0d, 0.0d);
    double score2 = model.scoreItem(fv2);
    assertEquals(8.0d + 5.0d + (1.0d + 2.0d * (0.0d + 10.0d) / 15.0d), score2, 0.001d);

    FeatureVector fv3 = makeFeatureVector(4.0d, 1.0d);
    double score3 = model.scoreItem(fv3);
    assertEquals(8.0d - 20.0d + (1.0d + 2.0d * (1.0d + 10.0d) / 15.0d), score3, 0.001d);

    FeatureVector fv4 = makeFeatureVector(2.0d, 7.0d);
    double score4 = model.scoreItem(fv4);
    assertEquals(8.0d + 10.0d + (1.0d + 2.0d * (7.0d + 10.0d) / 15.0d), score4, 0.001d);
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
    record2.setFunctionForm(FunctionForm.SPLINE);
    record2.setFeatureFamily("spline_float");
    record2.setFeatureName("aaa");
    record2.setWeightVector(ws);
    record2.setMinVal(1.0);
    record2.setMaxVal(3.0);
    ModelRecord record3 = new ModelRecord();
    record3.setFunctionForm(FunctionForm.SPLINE);
    record3.setFeatureFamily("spline_string");
    record3.setFeatureName("bbb");
    record3.setWeightVector(ws);
    record3.setMinVal(1.0);
    record3.setMaxVal(2.0);
    ModelRecord record4 = new ModelRecord();
    record4.setFunctionForm(FunctionForm.LINEAR);
    record4.setFeatureFamily("linear_float");
    record4.setFeatureName("ccc");
    record4.setWeightVector(wl);
    ModelRecord record5 = new ModelRecord();
    record5.setFunctionForm(FunctionForm.LINEAR);
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
    FeatureVector featureVector = makeFeatureVector(2.0d, 7.0d);
    try {
      Optional<AbstractModel> model = ModelFactory.createFromReader(reader, registry);
      assertTrue(model.isPresent());
      double score = model.get().scoreItem(featureVector);
      assertEquals(8.0d + 10.0d + 15.0d, score, 0.001d);
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
    model.addFunction(registry.feature("spline_float", "aaa"),
                      new Spline(2.0d, 10.0d, 5), false);
    // add an existing feature with overwrite
    model.addFunction(registry.feature("linear_float", "ccc"),
                      new Linear(3.0d, 5.0d), true);
    // add a new feature
    model.addFunction(registry.feature("spline_float", "new"),
                      new Spline(2.0d, 10.0d, 5), false);

    Map<Feature, Function> weights = model.weights();
    for (Map.Entry<Feature, Function>  entry: weights.entrySet()) {

      String featureName = entry.getKey().name();
      String familyName = entry.getKey().family().name();
      Function func = entry.getValue();
      if (familyName.equals("spline_float")) {
        Spline spline = (Spline) func;
        if (featureName.equals("aaa")) {
          assertTrue(spline.getMaxVal() == 3.0d);
          assertTrue(spline.getMinVal() == 1.0d);
          assertTrue(spline.getWeights().length == 3);
        } else if (featureName.equals("new")) {
          assertTrue(spline.getMaxVal() == 10.0d);
          assertTrue(spline.getMinVal() == 2.0d);
          assertTrue(spline.getWeights().length == 5);
        }
       } else if(familyName.equals("linear_float") && featureName.equals("ccc")) {
        Linear linear = (Linear) func;
        assertTrue(linear.getWeights().length == 2);
        assertTrue(linear.getWeights()[0] == 0.0d);
        assertTrue(linear.getWeights()[1] == 0.0d);
        assertTrue(linear.getMinVal() == 3.0d);
        assertTrue(linear.getMaxVal() == 5.0d);
      }
    }
  }
}
