package com.airbnb.aerosolve.core.scoring;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.features.InputGenerator;
import com.airbnb.aerosolve.core.features.InputSchema;
import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.features.SimpleExample;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

@Slf4j
public class ModelScorerTest {
  private final FeatureRegistry registry = new FeatureRegistry();

  @Test
  public void rawProbability() throws Exception {
    ModelConfig incomeModel = ModelConfig.builder()
        .modelName("income.model")
        .configName("income_prediction.conf")
        .key("spline_model")
        .build();

    ModelScorer modelScorer = new ModelScorer(incomeModel);

    InputSchema inputSchema = new InputSchema();
    inputSchema.add(dataName1);
    inputSchema.add(dataName2);
    inputSchema.add(dataName3);
    inputSchema.finish();

    InputGenerator f = new InputGenerator(inputSchema);
    f.add(data1, dataName1);
    f.add(data2, dataName2);
    f.add(data3, dataName3);

    Example example = new SimpleExample(registry);
    MultiFamilyVector featureVector = f.load(example.createVector());

    Family floatFeatureFamily = registry.family("F");
    assertEquals(featureVector.get(floatFeatureFamily.feature("age")), 30, 0.1);
    assertEquals(featureVector.get(floatFeatureFamily.feature("hours")), 40, 0.1);

    Family stringFeatureFamily = registry.family("S");
    assertFalse(featureVector.containsKey(stringFeatureFamily.feature("marital-status")));
    assertTrue(featureVector.containsKey(stringFeatureFamily.feature("marital-status:married")));

    double score = modelScorer.score(example);

    log.info("score {}", score);
  }

  private static final String[] dataName1 = {"F_age", "F_fnlwgt", "F_edu-num"};
  private static final double[] data1 = {30, 10, 10};

  private static final String[] dataName2 = {"F_capital-gain", "F_capital-loss", "F_hours"};
  private static final double[] data2 = {3000, 1000, 40};
  private static final String[] dataName3 = {
      "S_workclass", "S_education", "S_marital-status",
      "S_occupation", "S_relationship", "S_race", "S_sex",
      "S_native-country"
  };

  private static final String[] data3 = {
      "scientist", "collage", "married",
      "engineer", "single", "asian", "female",
      "usa"
  };

}