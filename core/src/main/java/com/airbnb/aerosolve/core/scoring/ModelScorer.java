package com.airbnb.aerosolve.core.scoring;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.models.AbstractModel;
import com.airbnb.aerosolve.core.models.ModelFactory;
import com.airbnb.aerosolve.core.transforms.Transformer;
import com.google.common.base.Optional;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import lombok.extern.slf4j.Slf4j;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

@Slf4j
public class ModelScorer {
  private final AbstractModel model;
  private final Transformer transformer;

  public ModelScorer(BufferedReader reader, ModelConfig model) throws IOException {
    Optional<AbstractModel> modelOpt = ModelFactory.createFromReader(reader);
    this.model = modelOpt.get();

    Config modelConfig = ConfigFactory.load(model.getConfigName());
    this.transformer = new Transformer(modelConfig, model.getKey());
  }

  /*
    this assumes model file in resource folder, i.e. test/resources/ in unit test
   */
  public ModelScorer(ModelConfig model) throws IOException {
    this(new BufferedReader(new InputStreamReader(
            ModelScorer.class.getResourceAsStream("/" + model.getModelName()))),
        model);
  }

  public double rawProbability(Example example) {
    return model.scoreProbability(score(example));
  }

  public float score(Example example) {
    FeatureVector featureVector = example.getExample().get(0);
    transformer.combineContextAndItems(example);
    return model.scoreItem(featureVector);
  }
}
