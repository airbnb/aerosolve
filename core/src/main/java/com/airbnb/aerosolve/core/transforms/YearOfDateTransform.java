package com.airbnb.aerosolve.core.transforms;

import java.util.Calendar;

/**
 * Get the year of date
 */
public class YearOfDateTransform extends DateTransform {
  @Override
  public Double doDateTransform(Calendar cal) {
    return (double)cal.get(Calendar.YEAR);
  }
}
