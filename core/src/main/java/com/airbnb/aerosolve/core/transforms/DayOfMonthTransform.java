package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Get day of month from date
 */
public class DayOfMonthTransform extends Transform {
  private String fieldName1;
  private String outputName;
  private final static SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");
  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    outputName = config.getString(key + ".output");
  }

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

    try {
      for (String dateStr: feature1) {
        Date date = format.parse(dateStr);
        Calendar cal = Calendar.getInstance();
        cal.setTime(date);
        // First day of the month is 1
        output.put(dateStr, (double)cal.get(Calendar.DAY_OF_MONTH));
      }
    } catch (ParseException e) {
      e.printStackTrace();
      return ;
    }

    floatFeatures.put(outputName, output);
  }
}
