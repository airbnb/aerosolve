package com.airbnb.aerosolve.core.online;

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

@Slf4j
public class ModelDAO {
  private final AbstractModel model;
  private final Transformer transformer;
  protected final FeatureMapping featureMapping;

  public ModelDAO(AbstractModel model,
                  FeatureMapping featureMapping,
                  Transformer transformer) {
    this.model = model;
    this.featureMapping = featureMapping;
    this.transformer = transformer;
  }

  public ModelDAO(FeatureMapping featureMapping,
                  BufferedReader reader, ModelType modelType) throws IOException {
    Optional<AbstractModel> modelOpt = ModelFactory.createFromReader(reader);
    this.model = modelOpt.get();

    Config modelConfig = ConfigFactory.load(modelType.getConfigName());
    this.transformer = new Transformer(modelConfig, modelType.getKey());
    this.featureMapping = featureMapping;
  }

  public double rawProbability(Example example) {
    FeatureVector featureVector = example.getExample().get(0);

    transformer.combineContextAndItems(example);

    final float score = model.scoreItem(featureVector);
    return model.scoreProbability(score);
  }
}
