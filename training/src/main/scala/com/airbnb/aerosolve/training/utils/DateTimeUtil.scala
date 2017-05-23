package com.airbnb.aerosolve.training.utils

import org.joda.time.format.{DateTimeFormat, DateTimeFormatter}

object DateTimeUtil {
  def dateMinus(date: String, days: Int): String = {
    // return a string for the date which is 'days' earlier than 'date'
    // e.g. dateMinus("2015-06-01", 1) returns "2015-05-31"
    val formatter : DateTimeFormatter = DateTimeFormat.forPattern("yyyy-MM-dd")
    val dateFmt = formatter.parseDateTime(date)
    formatter.print(dateFmt.minusDays(days))
  }

}
