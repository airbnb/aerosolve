package com.airbnb.aerosolve.core.transforms;


import java.util.Calendar;

/**
 * Get the day of year from date
 */
public class DayOfYearTransform extends DateTransform {
  @Override
  public Double doDateTransform(Calendar cal) {
    // First day of year is 1
    return (double)cal.get(Calendar.DAY_OF_YEAR);
  }
}
