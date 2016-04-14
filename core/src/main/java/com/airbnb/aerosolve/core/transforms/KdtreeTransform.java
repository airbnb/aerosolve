package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.DualFeatureTransform;
import com.airbnb.aerosolve.core.models.KDTreeModel;
import com.airbnb.aerosolve.core.KDTreeNode;
import com.google.common.base.Optional;
import com.typesafe.config.Config;

import java.util.ArrayList;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;
import lombok.extern.slf4j.Slf4j;

import javax.validation.constraints.NotNull;

/**
 * Inputs = fieldName1 (value1, value2)
 * Outputs = list of kdtree nodes and the distance from the split
 * This is the continuous version of the kd-tree transform and encodes
 * the distance from each splitting plane to the point being queried.
 * One can think of this as a tree kernel transform of a point.
 */
@Slf4j
@LegacyNames("kdtree_continuous")
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class KdtreeTransform extends DualFeatureTransform<KdtreeTransform> {
  private Integer maxCount;
  @NotNull
  private String modelEncoded;
  private boolean continuous;

  @Setter(AccessLevel.NONE)
  private Optional<KDTreeModel> modelOptional;

  @Override
  public KdtreeTransform configure(Config config, String key) {
    return super.configure(config, key)
        .maxCount(intFromConfig(config, key, ".max_count"))
        .modelEncoded(stringFromConfig(config, key, ".model_base64"))
        .continuous(isContinuous(config, key));
  }

  private boolean isContinuous(Config config, String key) {
    if (!booleanFromConfig(config, key, ".continuous")) {
      String transformType = getTransformType(config, key);
      // For legacy transform types.
      return transformType != null && transformType.endsWith("continuous");
    }
    return true;
  }

  @Override
  protected void setup() {
    super.setup();
    modelOptional = KDTreeModel.readFromGzippedBase64String(modelEncoded);
    if (!modelOptional.isPresent()) {
      String message = "Could not load KDTree from encoded field";
      log.error(message);
      throw new IllegalStateException(message);
    }
  }

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    double v1 = featureVector.getDouble(inputFeature);
    double v2 = featureVector.getDouble(otherFeature);

    ArrayList<Integer> result = modelOptional.get().query(v1, v2);
    int count = Math.min(result.size(), maxCount);

    if (continuous) {
      KDTreeNode[] nodes = modelOptional.get().nodes();

      for (int i = 0; i < count; i++) {
        Integer res = result.get(result.size() - 1 - i);
        double split = nodes[res].getSplitValue();
        switch (nodes[res].getNodeType()) {
          case X_SPLIT: {
            featureVector.put(outputFamily.feature(res.toString()), v1 - split);
          }
          break;
          case Y_SPLIT: {
            featureVector.put(outputFamily.feature(res.toString()), v2 - split);
          }
          break;
        }
      }
    } else {
      for (int i = 0; i < count; i++) {
        Integer res = result.get(result.size() - 1 - i);
        featureVector.putString(outputFamily.feature(res.toString()));
      }
    }
  }
}
