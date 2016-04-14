package com.airbnb.aerosolve.core.transforms;


import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.DualFamilyTransform;
import com.airbnb.aerosolve.core.util.Util;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.joda.time.DateTime;
import org.joda.time.Duration;

/**
 * output = date_diff(field1, field2)
 * get the date difference between dates in features of key "field1" and
 * dates in features of key "field2"
 */
// TODO (Brad): Configurable date time format.
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class DateDiffTransform extends DualFamilyTransform<DateDiffTransform> {

  @Override
  protected void doTransform(MultiFamilyVector featureVector) {

    // TODO (Brad): I made a mistake when refactoring this because it was unintuitive that the date
    // in field1 is the end date.  Is that intentional?
    try {
      for (FeatureValue value : getInput(featureVector)) {
        String endDateStr = value.feature().name();
        DateTime endDate = Util.DATE_FORMAT.parseDateTime(endDateStr);
        for (FeatureValue value2 : featureVector.get(otherFamily)) {
          String startDateStr = value2.feature().name();
          DateTime startDate = Util.DATE_FORMAT.parseDateTime(startDateStr);
          Duration duration = new Duration(startDate, endDate);
          Feature feature = outputFamily.feature(endDateStr + "-m-" + startDateStr);
          featureVector.put(feature, duration.getStandardDays());
        }
      }
    } catch (IllegalArgumentException e) {
      // TODO (Brad): Better error handling
      e.printStackTrace();
    }
  }
}
