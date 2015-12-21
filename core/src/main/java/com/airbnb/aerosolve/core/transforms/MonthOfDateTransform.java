package com.airbnb.aerosolve.core.transforms;

import java.util.Calendar;

/**
 * Get the month from date
 */
public class MonthOfDateTransform extends DateTransform {
  @Override
  public Double doDateTransform(Calendar cal) {
    // First month of the year is 1
    return (double)(cal.get(Calendar.MONTH) + 1);
  }
}
