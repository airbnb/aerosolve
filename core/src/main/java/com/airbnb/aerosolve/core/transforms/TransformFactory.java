package com.airbnb.aerosolve.core.transforms;

import com.google.common.base.CaseFormat;
import com.typesafe.config.Config;

/**
 * Created by hector_yee on 8/25/14.
 */
public class TransformFactory {
  public static Transform createTransform(Config config, String key) {
    if (config == null || key == null) {
      return null;
    }
    String transformName = config.getString(key + ".transform");
    if (transformName == null) {
      return null;
    }

    String name = CaseFormat.LOWER_UNDERSCORE.to(CaseFormat.UPPER_CAMEL, transformName);
    Transform result;
    try {
      result = (Transform) Class.forName("com.airbnb.aerosolve.core.transforms." + name + "Transform").newInstance();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    result.configure(config, key);
    return result;
  }
}
