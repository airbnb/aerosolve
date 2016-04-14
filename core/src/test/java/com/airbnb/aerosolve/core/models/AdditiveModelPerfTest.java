package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.perf.FastMultiFamilyVector;
import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.SimpleExample;
import com.airbnb.aerosolve.core.transforms.Transformer;
import com.airbnb.aerosolve.core.util.Util;
import com.google.common.base.Optional;
import com.google.common.collect.Iterables;
import com.google.common.io.ByteStreams;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import org.apache.commons.codec.binary.Base64;
import org.junit.Test;

/**
 *
 */
public class AdditiveModelPerfTest {
  private final FeatureRegistry registry = new FeatureRegistry();
  private final Random random = new Random();

  @Test
  public void testPerformance() throws Exception {

    ClassLoader loader = Thread.currentThread().getContextClassLoader();
    byte[] vectorBytes = ByteStreams.toByteArray(loader.getResourceAsStream("vector.bin"));
    String vectorStr = new String(Base64.encodeBase64(vectorBytes));
    FastMultiFamilyVector fastVector = (FastMultiFamilyVector) Util.decodeFeatureVector(vectorStr,
                                                                                        registry);
    List<Feature> features = new ArrayList<>(fastVector.keySet());
    InputStream modelStream = loader.getResourceAsStream("daphne.model");
    BufferedReader fileReader = new BufferedReader(new InputStreamReader(modelStream));
    Optional<AbstractModel> modelOpt = ModelFactory.createFromReader(fileReader, registry);
    if (!modelOpt.isPresent()) {
      throw new IllegalStateException("Could not load model");
    }
    AdditiveModel model = (AdditiveModel) modelOpt.get();

    Config config = ConfigFactory.parseResources("model_daphne.conf");
    Transformer transformer = new Transformer(config, "pricing_model_config", registry, model);

    double transformTime = 0;
    double scoreTime = 0;
    double scoreNum = 0;
    int iterations = 100;
    for (int i = 0; i < iterations; i++) {
      Example newExample = newExample(fastVector, features);

      long millis = System.nanoTime();
      newExample.transform(transformer, model);
      transformTime += System.nanoTime() - millis;
      millis = System.nanoTime();

      for (FeatureVector scoreVector : newExample) {
        scoreNum += model.scoreItem(scoreVector);
      }
      scoreTime += System.nanoTime() - millis;
    }
    transformTime = transformTime / 1000000;
    System.out.println(String.format("Took %.3f s to transform", transformTime/1000));
    System.out.println(String.format("Took %.2f micros per transform", transformTime/iterations));

    scoreTime = scoreTime / 1000000;
    System.out.println(String.format("Took %.3f s to score", scoreTime/1000));
    System.out.println(String.format("Took %.2f micros per score", scoreTime/iterations));

    System.out.println("ScoreNum " + scoreNum);
  }

  private Example newExample(FastMultiFamilyVector fastVector, List<Feature> features) {
    Example example = new SimpleExample(registry);
    for (int i = 0; i < 1000; i++) {
      fastVector = new FastMultiFamilyVector(fastVector);
      int index = random.nextInt(features.size());
      Feature feature = features.get(index);
      fastVector.put(feature, random.nextFloat());
      example.addToExample(fastVector);
    }
    return example;
  }
}
