package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

import java.util.HashMap;
import java.util.Map;

import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class FloatLabelTransformTest {
  private FeatureVector makeFeatureVector(Double value, Double label) {
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    if (value != null) {
      Map<String, Double> map = new HashMap<>();
      map.put("50th", value);
      floatFeatures.put("DECILES", map);
    }

    if (label != null) {
      Map<String, Double> map = new HashMap<>();
      map.put("", label);
      floatFeatures.put("LABEL", map);
    }

    FeatureVector featureVector = new FeatureVector();
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  private FloatLabelTransform createTransform(FloatLabelTransform.MergeStrategy mergeStrategy) {
    Config config = ConfigFactory.parseString("test_float_label {\n" +
        " transform : float_label\n" +
        " field1 : DECILES\n" +
        " key1 : 50th\n" +
        " threshold : 10\n" +
        " merge : " + mergeStrategy.toString() + "\n" +
        "}");

    FloatLabelTransform transform = new FloatLabelTransform();
    transform.configure(config, "test_float_label");

    return transform;
  }

  private double getLabel(FeatureVector featureVector) {
    return featureVector.floatFeatures.get("LABEL").get("");
  }

  @Test
  public void handleMissingFeature() {
    FloatLabelTransform transform = createTransform(FloatLabelTransform.MergeStrategy.OVERRIDE);

    // keep current label even though we could overwrite
    FeatureVector featureVector1 = makeFeatureVector(null, 1.0);
    transform.doTransform(featureVector1);
    assertEquals(1, getLabel(featureVector1), 0.1);
  }

  @Test
  public void OverwriteLabel() {
    FloatLabelTransform transform = createTransform(FloatLabelTransform.MergeStrategy.OVERRIDE);

    // fill in missing positive label
    FeatureVector featureVector1 = makeFeatureVector(1.0, null);
    transform.doTransform(featureVector1);
    assertEquals(-1, getLabel(featureVector1), 0.1);

    // fill in missing negative label
    FeatureVector featureVector2 = makeFeatureVector(10.0, null);
    transform.doTransform(featureVector2);
    assertEquals(1, getLabel(featureVector2), 0.1);

    // overwrite positive label
    FeatureVector featureVector3 = makeFeatureVector(1.0, 1.0);
    transform.doTransform(featureVector3);
    assertEquals(-1, getLabel(featureVector3), 0.1);

    // overwrite negative label
    FeatureVector featureVector4 = makeFeatureVector(10.0, -1.0);
    transform.doTransform(featureVector4);
    assertEquals(1, getLabel(featureVector4), 0.1);
  }

  @Test
  public void keepExistingLabel() {
    FloatLabelTransform transform = createTransform(FloatLabelTransform.MergeStrategy.SKIP);

    // fill in missing positive label
    FeatureVector featureVector1 = makeFeatureVector(1.0, null);
    transform.doTransform(featureVector1);
    assertEquals(-1, getLabel(featureVector1), 0.1);

    // fill in missing negative label
    FeatureVector featureVector2 = makeFeatureVector(10.0, null);
    transform.doTransform(featureVector2);
    assertEquals(1, getLabel(featureVector2), 0.1);

    // preserve positive label
    FeatureVector featureVector3 = makeFeatureVector(1.0, 1.0);
    transform.doTransform(featureVector3);
    assertEquals(1, getLabel(featureVector3), 0.1);

    // preserve negative label
    FeatureVector featureVector4 = makeFeatureVector(10.0, -1.0);
    transform.doTransform(featureVector4);
    assertEquals(-1, getLabel(featureVector4), 0.1);
  }

  @Test
  public void keepPositiveLabel() {
    FloatLabelTransform transform = createTransform(FloatLabelTransform.MergeStrategy.OVERRIDE_NEGATIVE);

    // fill in missing positive label
    FeatureVector featureVector1 = makeFeatureVector(1.0, null);
    transform.doTransform(featureVector1);
    assertEquals(-1, getLabel(featureVector1), 0.1);

    // fill in missing negative label
    FeatureVector featureVector2 = makeFeatureVector(10.0, null);
    transform.doTransform(featureVector2);
    assertEquals(1, getLabel(featureVector2), 0.1);

    // preserve positive label
    FeatureVector featureVector3 = makeFeatureVector(1.0, 1.0);
    transform.doTransform(featureVector3);
    assertEquals(1, getLabel(featureVector3), 0.1);

    // overwrite negative label
    FeatureVector featureVector4 = makeFeatureVector(10.0, -1.0);
    transform.doTransform(featureVector4);
    assertEquals(1, getLabel(featureVector4), 0.1);
  }

  @Test
  public void keepNegativeLabel() {
    FloatLabelTransform transform = createTransform(FloatLabelTransform.MergeStrategy.OVERRIDE_POSITIVE);

    // fill in missing positive label
    FeatureVector featureVector1 = makeFeatureVector(1.0, null);
    transform.doTransform(featureVector1);
    assertEquals(-1, getLabel(featureVector1), 0.1);

    // fill in missing negative label
    FeatureVector featureVector2 = makeFeatureVector(10.0, null);
    transform.doTransform(featureVector2);
    assertEquals(1, getLabel(featureVector2), 0.1);

    // overwrite positive label
    FeatureVector featureVector3 = makeFeatureVector(1.0, 1.0);
    transform.doTransform(featureVector3);
    assertEquals(-1, getLabel(featureVector3), 0.1);

    // preserve negative label
    FeatureVector featureVector4 = makeFeatureVector(10.0, -1.0);
    transform.doTransform(featureVector4);
    assertEquals(-1, getLabel(featureVector4), 0.1);
  }
}
