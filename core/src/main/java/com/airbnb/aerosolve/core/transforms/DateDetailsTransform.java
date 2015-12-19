package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Return (year, month, dayofmonth, dayofweek) of a date
 */
public class DateDetailsTransform extends Transform {
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
        output.put(dateStr + "-year", (double)cal.get(Calendar.YEAR));
        // First month of the year is 1
        output.put(dateStr + "-month", (double)cal.get(Calendar.MONTH) + 1);
        // First day of month has value 1
        output.put(dateStr + "-dayofmonth", (double)cal.get(Calendar.DAY_OF_MONTH));
        /*
         SUNDAY - SATURDAY : 1 - 7,
        */
        output.put(dateStr + "-dayofweek", (double)cal.get(Calendar.DAY_OF_WEEK));
      }
    } catch (ParseException e) {
      e.printStackTrace();
      return ;
    }

    floatFeatures.put(outputName, output);
  }
}
