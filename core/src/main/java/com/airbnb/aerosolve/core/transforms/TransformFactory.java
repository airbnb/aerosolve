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

    // TODO: remove this once all configs have migrated over to the new transform names
    if (transformName.equals("delete_string_feature_column")) {
      transformName = "delete_string_feature_family";
    }

    String name = CaseFormat.LOWER_UNDERSCORE.to(CaseFormat.UPPER_CAMEL, transformName);
    Transform result = null;
    try {
      result = (Transform) Class.forName("com.airbnb.aerosolve.core.transforms." + name + "Transform").newInstance();
      result.configure(config, key);
    } catch (InstantiationException e) {
      e.printStackTrace();
    } catch (IllegalAccessException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }
    return result;
  }
}
