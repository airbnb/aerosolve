package com.airbnb.aerosolve.core.scoring;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.models.AbstractModel;
import com.airbnb.aerosolve.core.models.ModelFactory;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.transforms.Transformer;
import com.google.common.base.Optional;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class ModelScorer {
  private final AbstractModel model;
  private final Transformer transformer;

  public ModelScorer(BufferedReader reader, ModelConfig model, FeatureRegistry registry)
      throws IOException {
    Optional<AbstractModel> modelOpt = ModelFactory.createFromReader(reader, registry);
    this.model = modelOpt.get();

    Config modelConfig = ConfigFactory.load(model.getConfigName());
    this.transformer = new Transformer(modelConfig, model.getKey(), registry, modelOpt.get());
  }

  /*
    this assumes model file in resource folder, i.e. test/resources/ in unit test
   */
  public ModelScorer(ModelConfig model, FeatureRegistry registry) throws IOException {
    this(new BufferedReader(new InputStreamReader(
            ModelScorer.class.getResourceAsStream("/" + model.getModelName()))),
        model, registry);
  }

  public double rawProbability(Example example) {
    return model.scoreProbability(score(example));
  }

  public double score(Example example) {
    example.transform(transformer);
    FeatureVector featureVector = example.iterator().next();
    return model.scoreItem(featureVector);
  }
}
