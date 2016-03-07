package com.airbnb.aerosolve.core.util;

import lombok.extern.slf4j.Slf4j;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

@Slf4j
public class DateUtil {
  public static <R> Map<String, R> applyForDateRange(String startDate, String endDate, Function<String, R> f) {
    SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd");
    Map<String, R> map = new HashMap<>();
    try {
      Date startD = formatter.parse(startDate);
      Date endD = formatter.parse(endDate);
      LocalDate start = startD.toInstant().atZone(ZoneId.systemDefault()).toLocalDate();
      LocalDate end = endD.toInstant().atZone(ZoneId.systemDefault()).toLocalDate();

      for (LocalDate date = start; date.isBefore(end); date = date.plusDays(1)) {
        String d = date.toString();
        R r = f.apply(d);
        if (r != null) {
          map.put(d, r);
        }
      }
    } catch (ParseException e) {
      log.error("wrong date format? {} {}", startDate, endDate);
    }
    return map;
  }

}
