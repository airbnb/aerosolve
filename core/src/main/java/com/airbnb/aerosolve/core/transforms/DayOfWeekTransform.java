package com.airbnb.aerosolve.core.transforms;

import java.util.Calendar;

/**
 * Get the day of week from date
 */
public class DayOfWeekTransform extends DateTransform {
  @Override
  public Double doDateTransform(Calendar cal) {
    // SUNDAY - SATURDAY : 1 - 7,
    return (double)cal.get(Calendar.DAY_OF_WEEK);
  }
}
