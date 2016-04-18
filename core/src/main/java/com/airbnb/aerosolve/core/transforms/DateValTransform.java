package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import com.typesafe.config.Config;

import java.text.SimpleDateFormat;
import java.util.*;
import java.text.ParseException;

/**
 * Get the date value from date string
 * "field1" specifies the key of feature
 * "field2" specifies the type of date value
 */
public class DateValTransform implements Transform {
  protected String fieldName1;
  protected String dateType;
  protected String outputName;
  protected final static SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");
  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    dateType = config.getString(key + ".date_type");
    outputName = config.getString(key + ".output");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    if (stringFeatures == null) {
      return ;
    }

    Set<String> feature1 = stringFeatures.get(fieldName1);
    if (feature1 == null) {
      return ;
    }

    Util.optionallyCreateFloatFeatures(featureVector);
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    Map<String, Double> output = Util.getOrCreateFloatFeature(outputName, floatFeatures);

    for (String dateStr: feature1) {
      try {
        Date date = format.parse(dateStr);
        Calendar cal = Calendar.getInstance();
        cal.setTime(date);
        double dateVal;
        switch (dateType) {
          case "day_of_month":
            dateVal = cal.get(Calendar.DAY_OF_MONTH);
            break;
          case "day_of_week":
            dateVal = cal.get(Calendar.DAY_OF_WEEK);
            break;
          case "day_of_year":
            dateVal = cal.get(Calendar.DAY_OF_YEAR);
            break;
          case "year":
            dateVal = cal.get(Calendar.YEAR);
            break;
          case "month":
            dateVal = cal.get(Calendar.MONTH) + 1;
            break;
          default:
            return ;
        }
        output.put(dateStr, dateVal);
      } catch (ParseException e) {
        e.printStackTrace();
        continue ;
      }
    }
  }
}
