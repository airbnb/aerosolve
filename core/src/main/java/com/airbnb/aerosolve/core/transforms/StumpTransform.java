package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.perf.Family;
import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.types.ConfigurableTransform;
import com.typesafe.config.Config;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Setter;
import lombok.Value;
import lombok.experimental.Accessors;
import org.hibernate.validator.constraints.NotEmpty;

import javax.validation.constraints.NotNull;

/**
 * Applies boosted stump transform to float features and places them in string feature output.
 * The format for a stump feature family, feature name, threshold, descriptive name
 * You can obtain the stumps from a BoostedStump model in spark shell using
 * val model = sc.textFile(name).map(Util.decodeModel).take(10).map(x =&gt;
 *  "%s,%s,%f".format(x.featureFamily,x.featureName,x.threshold)).foreach(println)
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
public class StumpTransform extends ConfigurableTransform<StumpTransform> {
  @NotNull
  private String outputFamilyName;
  @NotNull
  @NotEmpty
  private List<String> stumps;

  // Family -> description
  @Setter(AccessLevel.NONE)
  private List<StumpDescription> thresholds;
  @Setter(AccessLevel.NONE)
  private Family outputFamily;

  @Override
  public StumpTransform configure(Config config, String key) {
    return outputFamilyName(stringFromConfig(config, key, ".output"))
        .stumps(stringListFromConfig(config, key,  ".stumps", true));
  }

  @Override
  protected void setup() {
    super.setup();
    outputFamily = registry.family(outputFamilyName);
    thresholds = new ArrayList<>(stumps.size());
    for (String stump : stumps) {
      String[] tokens = stump.split(",");
      if (tokens.length == 4) {
        String family = tokens[0];
        String featureName = tokens[1];
        Feature feature = registry.feature(family, featureName);
        double threshold = Double.parseDouble(tokens[2]);
        String descriptiveName = tokens[3];
        Feature outputFeature = outputFamily.feature(descriptiveName);
        StumpDescription description = new StumpDescription(feature,
                                                            threshold,
                                                            outputFeature);
        thresholds.add(description);
      }
    }
  }

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    for (StumpDescription stump : thresholds) {
      if (!featureVector.containsKey(stump.feature())) {
        continue;
      }

      double value = featureVector.getDouble(stump.feature());
      if (value >= stump.threshold()) {
        featureVector.putString(stump.outputFeature());
      }
    }
  }

  @Value
  private static class StumpDescription implements Serializable {
    private final Feature feature;
    private final double threshold;
    private final Feature outputFeature;
  }
}
