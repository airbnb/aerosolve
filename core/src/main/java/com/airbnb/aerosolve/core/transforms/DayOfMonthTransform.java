package com.airbnb.aerosolve.core.transforms;


import java.util.Calendar;

/**
 * Get day of month from date
 */
public class DayOfMonthTransform extends DateTransform {
  @Override
  public Double doDateTransform(Calendar cal) {
    // First day of month is 1
    return (double)cal.get(Calendar.DAY_OF_MONTH);
  }
}
