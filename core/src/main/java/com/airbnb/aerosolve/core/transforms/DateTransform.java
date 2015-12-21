package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;

import java.text.SimpleDateFormat;
import java.util.*;
import java.text.ParseException;

/**
 * Created by seckcoder on 12/20/15.
 */
public abstract class DateTransform extends Transform {
  protected String fieldName1;
  protected String outputName;
  protected final static SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");
  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    outputName = config.getString(key + ".output");
  }

  abstract public Double doDateTransform(Calendar cal);

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    if (stringFeatures == null || floatFeatures == null) {
      return ;
    }

    Set<String> feature1 = stringFeatures.get(fieldName1);
    if (feature1 == null) {
      return ;
    }

    Map<String, Double> output = new HashMap<>();

    for (String dateStr: feature1) {
      try {
        Date date = format.parse(dateStr);
        Calendar cal = Calendar.getInstance();
        cal.setTime(date);
        output.put(dateStr, doDateTransform(cal));
      } catch (ParseException e) {
        e.printStackTrace();
        continue ;
      }
    }

    floatFeatures.put(outputName, output);
  }
}
