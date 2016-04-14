package com.airbnb.aerosolve.core.scoring;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.features.FeatureGen;
import com.airbnb.aerosolve.core.features.FeatureMapping;
import com.airbnb.aerosolve.core.features.FeatureVectorGen;
import com.airbnb.aerosolve.core.features.Features;
import com.airbnb.aerosolve.core.features.FloatFamily;
import com.airbnb.aerosolve.core.features.StringFamily;
import com.airbnb.aerosolve.core.perf.Family;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import java.util.ArrayList;
import java.util.List;
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

    ModelScorer modelScorer = new ModelScorer(incomeModel, registry);

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

    Example example = FeatureVectorGen.toSingleFeatureVectorExample(features, stringFamilies,
                                                                    floatFamilies, registry);

    MultiFamilyVector featureVector = example.iterator().next();
    Family floatFeatureFamily = registry.family("F");
    assertEquals(featureVector.get(floatFeatureFamily.feature("age")), 30, 0.1);
    assertEquals(featureVector.get(floatFeatureFamily.feature("hours")), 40, 0.1);

    Family stringFeatureFamily = registry.family("S");
    assertFalse(featureVector.containsKey(stringFeatureFamily.feature("marital-status")));
    assertTrue(featureVector.containsKey(stringFeatureFamily.feature("marital-status:married")));

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