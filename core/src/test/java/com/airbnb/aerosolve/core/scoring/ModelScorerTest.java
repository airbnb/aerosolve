package com.airbnb.aerosolve.core.scoring;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.features.*;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

@Slf4j
public class ModelScorerTest {

  @Test
  public void rawProbability() throws Exception {
    ModelConfig incomeModel = ModelConfig.builder()
        .modelName("income.model")
        .configName("income_prediction.conf")
        .key("spline_model")
        .build();

    ModelScorer modelScorer = new ModelScorer(incomeModel);

    FeatureMapping featureMapping = new FeatureMapping();
    featureMapping.add(dataName1);
    featureMapping.add(dataName2);
    featureMapping.add(dataName3);
    featureMapping.finish();

    FeatureGen f = new FeatureGen(featureMapping);
    f.add(data1, dataName1);
    f.add(data2, dataName2);
    f.add(data3, dataName3);
    Features features = f.gen();

    List<StringFamily> stringFamilies = new ArrayList<>();
    stringFamilies.add(new StringFamily("S"));

    List<FloatFamily> floatFamilies = new ArrayList<>();
    floatFamilies.add(new FloatFamily("F"));

    Example example = FeatureVectorGen.toSingleFeatureVectorExample(features, stringFamilies, floatFamilies);

    FeatureVector featureVector = example.getExample().get(0);
    final Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    Map<String, Double> floatFeatureFamily = floatFeatures.get("F");
    assertEquals(floatFeatureFamily.get("age"), 30, 0.1);
    assertEquals(floatFeatureFamily.get("hours"), 40, 0.1);

    final Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Set<String> stringFeatureFamily = stringFeatures.get("S");
    assertFalse(stringFeatureFamily.contains("marital-status"));
    assertTrue(stringFeatureFamily.contains("married"));

    double score = modelScorer.score(example);

    log.info("score {}", score);
  }

  private static final String[] dataName1 = {"age", "fnlwgt", "edu-num"};
  private static final float[] data1 = {30, 10, 10};

  private static final String[] dataName2 = {"capital-gain", "capital-loss", "hours"};
  private static final float[] data2 = {3000, 1000, 40};
  private static final String[] dataName3 = {
      "workclass", "education", "marital-status",
      "occupation", "relationship", "race", "sex",
      "native-country"
  };

  private static final String[] data3 = {
      "scientist", "collage", "married",
      "engineer", "single", "asian", "female",
      "usa"
  };

}