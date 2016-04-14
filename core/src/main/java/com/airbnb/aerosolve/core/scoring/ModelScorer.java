package com.airbnb.aerosolve.core.scoring;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.models.AbstractModel;
import com.airbnb.aerosolve.core.models.ModelFactory;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
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
  private final Feature biasFeature;

  public ModelScorer(BufferedReader reader, ModelConfig modelConfig)
      throws IOException {
    // Let the model create the FeatureRegistry because we're using one Transformer per Model in
    // this class.
    FeatureRegistry registry = new FeatureRegistry();
    Optional<AbstractModel> modelOpt = ModelFactory.createFromReader(reader, registry);
    if (!modelOpt.isPresent()) {
      throw new IllegalStateException("Could not create model from reader");
    }
    this.model = modelOpt.get();

    Config transformerConfig = ConfigFactory.load(modelConfig.getConfigName());
    this.transformer = new Transformer(transformerConfig, modelConfig.getKey(),
                                       this.model.registry(), this.model);
    this.biasFeature = this.model.registry().feature("BIAS", "B");
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

  public double score(Example example) {
    // TODO (Brad): This is really kind of odd. Why have an example if we assume it's one item?!
    FeatureVector featureVector = example.iterator().next();

    // TODO (Brad): Maybe this should be a part of the transform itself.  Why is it important? Does
    // everyone want it?
    featureVector.putString(biasFeature);

    // TODO (Brad): Abusing the fact that transforms are in place.  Let's fix this too.
    example.transform(transformer);

    return model.scoreItem(featureVector);
  }
}
