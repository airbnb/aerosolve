package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.models.DecisionTreeModel;
import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.ConfigurableTransform;
import com.typesafe.config.Config;
import java.util.List;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;
import org.hibernate.validator.constraints.NotEmpty;

import javax.validation.constraints.NotNull;

/**
 * Applies a decision tree transform to existing float features.
 * Emits the binary leaf features to the string family output_leaves
 * Emits the score to the float family output_score
 * Use tree.toHumanReadableTransform to generate the nodes list.
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class DecisionTreeTransform extends ConfigurableTransform<DecisionTreeTransform> {
  @NotNull
  private String outputLeavesFamilyName;
  @NotNull
  private String outputScoreFamilyName;
  @NotNull
  private String outputScoreFeatureName;
  @NotNull
  @NotEmpty
  private List<String> nodes;

  @Setter(AccessLevel.NONE)
  private Family outputLeavesFamily;
  @Setter(AccessLevel.NONE)
  private Feature outputScoreFeature;
  @Setter(AccessLevel.NONE)
  private DecisionTreeModel tree;

  @Override
  public DecisionTreeTransform configure(Config config, String key) {
    return outputLeavesFamilyName(stringFromConfig(config, key, ".output_leaves"))
        .outputScoreFamilyName(stringFromConfig(config, key, ".output_score_family"))
        .outputScoreFeatureName(stringFromConfig(config, key, ".output_score_name"))
        .nodes(stringListFromConfig(config, key, ".nodes", true));
  }

  @Override
  protected void setup() {
    super.setup();
    outputLeavesFamily = registry.family(outputLeavesFamilyName);
    outputScoreFeature = registry.feature(outputScoreFamilyName, outputScoreFeatureName);
    tree = DecisionTreeModel.fromHumanReadableTransform(nodes, registry);
  }

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    int leafIdx = tree.getLeafIndex(featureVector);
    ModelRecord rec = tree.stumps().get(leafIdx);
    featureVector.put(outputLeavesFamily.feature(rec.getFeatureName()), 1.0d);
    featureVector.put(outputScoreFeature, rec.getFeatureWeight());
  }
}
