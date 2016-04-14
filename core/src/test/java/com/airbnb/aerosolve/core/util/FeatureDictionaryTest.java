package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.TransformTestingHelper;
import java.util.ArrayList;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
@Slf4j
public class FeatureDictionaryTest {
  private final FeatureRegistry registry = new FeatureRegistry();

  public MultiFamilyVector makeDenseFeatureVector(String id,
                                              double v1,
                                              double v2) {
    MultiFamilyVector vector = TransformTestingHelper.makeEmptyVector(registry);
    vector.putDense(registry.family("a"), new double[]{v1, v2});
    vector.putDense(registry.family("b"), new double[]{v2, v1});
    vector.putString(registry.feature("id", id));
    return vector;
  }

  public MultiFamilyVector makeSparseFeatureVector(String id,
                                               String v1,
                                               String v2) {
    MultiFamilyVector vector = TransformTestingHelper.makeEmptyVector(registry);
    Family stringFamily = registry.family("a");
    vector.putString(stringFamily.feature(v1));
    vector.putString(stringFamily.feature(v2));
    vector.putString(registry.feature("id", id));
    return vector;
  }

  @Test
  public void testDictionaryMinKernel() {
    List<MultiFamilyVector> dict = new ArrayList<>();
    dict.add(makeDenseFeatureVector("0", 0.3, 0.7));
    dict.add(makeDenseFeatureVector("1", 0.7, 0.3));
    dict.add(makeDenseFeatureVector("2", 0.5, 0.5));
    dict.add(makeDenseFeatureVector("3", 0.0, 0.0));
    FeatureDictionary dictionary = new MinKernelDenseFeatureDictionary(registry);
    dictionary.setDictionaryList(dict);

    Family outputFamily = registry.family("mk");
    KNearestNeighborsOptions opt = KNearestNeighborsOptions.builder()
        .idKey(registry.family("id"))
        .outputKey(outputFamily)
        .numNearest(2)
        .build();

    MultiFamilyVector response1 = dictionary.getKNearestNeighbors(opt, dict.get(0));
    log.info(response1.toString());
    assertTrue(response1.size() > 0);
    assertEquals(response1.get(outputFamily).size(), 2);
    assertEquals(response1.get(outputFamily.feature("1")), 1.2, 0.1);
    assertEquals(response1.get(outputFamily.feature("2")), 1.6, 0.1);

    MultiFamilyVector response2 = dictionary.getKNearestNeighbors(opt, dict.get(1));
    log.info(response2.toString());
    assertTrue(response2.size() > 0);
    assertEquals(response2.get(outputFamily).size(), 2);
    assertEquals(response2.get(outputFamily.feature("0")), 1.2, 0.1);
    assertEquals(response2.get(outputFamily.feature("2")), 1.6, 0.1);

    MultiFamilyVector response3 = dictionary.getKNearestNeighbors(opt, dict.get(2));
    log.info(response3.toString());
    assertTrue(response3.size() > 0);
    assertEquals(response3.get(outputFamily).size(), 2);
    assertEquals(response3.get(outputFamily.feature("0")), 1.6, 0.1);
    assertEquals(response3.get(outputFamily.feature("1")), 1.6, 0.1);
  }

  @Test
  public void testDictionaryLSH() {
    List<MultiFamilyVector> dict = new ArrayList<>();
    dict.add(makeSparseFeatureVector("0", "a", "b"));
    dict.add(makeSparseFeatureVector("1", "b", "c"));
    dict.add(makeSparseFeatureVector("2", "a", "e"));
    dict.add(makeSparseFeatureVector("3", "e", "f"));
    dict.add(makeSparseFeatureVector("4", "@@@", "$$$"));
    dict.add(makeSparseFeatureVector("5", "$$$", "@@@"));
    FeatureDictionary dictionary = new LocalitySensitiveHashSparseFeatureDictionary(registry);
    dictionary.setDictionaryList(dict);

    Family outputFamily = registry.family("sim");
    KNearestNeighborsOptions opt = KNearestNeighborsOptions.builder()
        .idKey(registry.family("id"))
        .outputKey(outputFamily)
        .featureKey(registry.family("a"))
        .numNearest(2)
        .build();

    MultiFamilyVector response1 = dictionary.getKNearestNeighbors(opt, dict.get(0));
    log.info(response1.toString());
    assertTrue(response1.size() > 0);
    assertEquals(response1.get(outputFamily).size(), 2);
    assertEquals(response1.get(outputFamily.feature("1")), 1.0, 0.1);
    assertEquals(response1.get(outputFamily.feature("2")), 1.0, 0.1);

    MultiFamilyVector response2 = dictionary.getKNearestNeighbors(opt, dict.get(1));
    log.info(response2.toString());
    assertTrue(response2.size() > 0);
    assertEquals(response2.get(outputFamily).size(), 1);
    assertEquals(response2.get(outputFamily.feature("0")), 1.0, 0.1);

    MultiFamilyVector response3 = dictionary.getKNearestNeighbors(opt, dict.get(2));
    log.info(response3.toString());
    assertTrue(response3.size() > 0);
    assertEquals(response3.get(outputFamily).size(), 2);
    assertEquals(response3.get(outputFamily.feature("0")), 1.0, 0.1);
    assertEquals(response3.get(outputFamily.feature("3")), 1.0, 0.1);

    MultiFamilyVector response4 = dictionary.getKNearestNeighbors(opt, dict.get(3));
    log.info(response4.toString());
    assertTrue(response4.size() > 0);
    assertEquals(response4.get(outputFamily).size(), 1);
    assertEquals(response4.get(outputFamily.feature("2")), 1.0, 0.1);

    MultiFamilyVector response5 = dictionary.getKNearestNeighbors(opt, dict.get(4));
    log.info(response5.toString());
    assertTrue(response5.size() > 0);
    assertEquals(response5.get(outputFamily).size(), 1);
    assertEquals(response5.get(outputFamily.feature("5")), 2.0, 0.1);
  }
}
