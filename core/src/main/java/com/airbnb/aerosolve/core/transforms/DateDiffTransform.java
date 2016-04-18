package com.airbnb.aerosolve.core.transforms;


import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import com.typesafe.config.Config;

import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * output = date_diff(field1, field2)
 * get the date difference between dates in features of key "field1" and
 * dates in features of key "field2"
 */
public class DateDiffTransform implements Transform {
  private String fieldName1;
  private String fieldName2;
  private String outputName;
  private final static SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    fieldName2 = config.getString(key + ".field2");
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
    Set<String> feature2 = stringFeatures.get(fieldName2);
    if (feature1 == null || feature2 == null) {
      return ;
    }

    Map<String, Double> output = Util.getOrCreateFloatFeature(outputName, floatFeatures);

    try {
      for (String endDateStr : feature1) {
        Date endDate = format.parse(endDateStr);
        for (String startDateStr : feature2) {
          Date startDate = format.parse(startDateStr);
          long diff = endDate.getTime() - startDate.getTime();
          long diffDays = TimeUnit.DAYS.convert(diff, TimeUnit.MILLISECONDS);
          output.put(endDateStr + "-m-" + startDateStr, (double)diffDays);
        }
      }
    } catch (ParseException e) {
      e.printStackTrace();
      return ;
    }
  }
}
