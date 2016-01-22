package com.airbnb.aerosolve.core.transforms;

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
    Transform result = null;
    switch (transformName) {
      case "cross": {
        result = new CrossTransform();
        break;
      }
      case "self_cross": {
        result = new SelfCrossTransform();
        break;
      }
      case "string_cross_float": {
        result = new StringCrossFloatTransform();
        break;
      }
      case "quantize": {
        result = new QuantizeTransform();
        break;
      }
      case "multiscale_quantize": {
        result = new MultiscaleQuantizeTransform();
        break;
      }
      case "multiscale_grid_quantize": {
        result = new MultiscaleGridQuantizeTransform();
        break;
      }
      case "custom_multiscale_quantize": {
        result = new CustomMultiscaleQuantizeTransform();
        break;
      }
      case "multiscale_grid_continuous" : {
        result = new MultiscaleGridContinuousTransform();
        break;
      }
      case "move_float_to_string": {
        result = new MoveFloatToStringTransform();
        break;
      }
      case "multiscale_move_float_to_string": {
        result = new MultiscaleMoveFloatToStringTransform();
        break;
      }
      case "custom_linear_log_quantize": {
        result = new CustomLinearLogQuantizeTransform();
        break;
      }
      case "linear_log_quantize": {
        result = new LinearLogQuantizeTransform();
        break;
      }
      case "list": {
        result = new ListTransform();
        break;
      }
      case "wta": {
        result = new WTAHashTransform();
        break;
      }
      case "delete_float_feature": {
        result = new DeleteFloatFeatureTransform();
        break;
      }
      case "delete_string_feature": {
        result = new DeleteStringFeatureTransform();
        break;
      }
      case "product" : {
        result = new ProductTransform();
        break;
      }
      case "subtract" : {
        result = new SubtractTransform();
        break;
      }
      case "divide" : {
        result = new DivideTransform();
        break;
      }
      case "nearest" : {
        result = new NearestTransform();
        break;
      }
      case "bucket_float" : {
        result = new BucketFloatTransform();
        break;
      }
      case "cap_float" : {
        result = new CapFloatFeatureTransform();
        break;
      }
      case "cut_float" : {
        result = new CutFloatFeatureTransform();
        break;
      }
      case "stuff_id" : {
        result = new StuffIdIntoFeatureTransform();
        break;
      }
      case "approximate_percentile" : {
        result = new ApproximatePercentileTransform();
        break;
      }
      case "kdtree" : {
        result = new KDTreeTransform();
        break;
      }
      case "kdtree_continuous" : {
        result = new KDTreeContinuousTransform();
        break;
      }
      case "stump" : {
        result = new StumpTransform();
        break;
      }
      case "decision_tree" : {
        result = new DecisionTreeTransform();
        break;
      }
      case "date_diff" : {
        result = new DateDiffTransform();
        break;
      }
      case "date_val" : {
        result = new DateTransform();
        break;
      }
      case "math_float" : {
        result = new FloatFeatureMathTransform();
        break;
      }
    }
    if (result != null) {
      result.configure(config, key);
    }
    return result;
  }
}
