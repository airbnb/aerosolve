package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.BaseFeaturesTransform;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.Accessors;
import lombok.extern.slf4j.Slf4j;
import org.joda.time.DateTime;
import org.joda.time.DateTimeFieldType;

import javax.validation.constraints.NotNull;

/**
 * Get the date value from date string
 * "field1" specifies the key of feature
 * "date_type" specifies the type of date value
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Slf4j
@Accessors(fluent = true, chain = true)
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class DateValTransform extends BaseFeaturesTransform<DateValTransform> {
  protected DateTimeFieldType dateTimeFieldType;

  @NotNull
  protected String dateType;

  @Override
  public DateValTransform configure(Config config, String key) {
    return super.configure(config, key)
        .dateType(stringFromConfig(config, key, ".date_type"));
  }

  @Override
  protected void setup() {
    super.setup();
    dateTimeFieldType = getDateTimeFieldType(dateType);
  }

  private DateTimeFieldType getDateTimeFieldType(String dateType) {
    switch (dateType) {
      case "day_of_month":
        return DateTimeFieldType.dayOfMonth();
      case "day_of_week":
        return DateTimeFieldType.dayOfWeek();
      case "day_of_year":
        return DateTimeFieldType.dayOfYear();
      case "year":
        return DateTimeFieldType.year();
      case "month":
        return DateTimeFieldType.monthOfYear();
      default:
        return null;
    }
  }

  @Override
  protected void doTransform(MultiFamilyVector featureVector) {
    for (FeatureValue value : getInput(featureVector)) {
      String dateStr = value.feature().name();
      try {
        DateTime date = Util.DATE_FORMAT.parseDateTime(dateStr);
        double dateVal = date.get(dateTimeFieldType);
        if (dateTimeFieldType.equals(DateTimeFieldType.dayOfWeek())) {
          // Joda DateTimes start the week with Monday.  So, Sunday is 7. We mod 7 to bring it to
          // 0 and add 1 to every day to offset.
          dateVal = (double) ((((int) dateVal) % 7) + 1);
        }
        featureVector.put(outputFamily.feature(dateStr), dateVal);
      } catch (IllegalArgumentException e) {
        log.error("Error parsing date String %s with format %s: %s",
                  dateStr, Util.DATE_FORMAT.toString(), e.getMessage());
        // Let's just continue here.  It doesn't seem worth aborting on a malformed String.
        // Hopefully someone checks the logs when this happens.
      }
    }
  }
}
