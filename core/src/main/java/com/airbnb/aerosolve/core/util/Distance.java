package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.KDTreeNode;

// http://www.geodatasource.com/developers/java
public class Distance {
  public static double miles(double lat1, double lon1, double lat2, double lon2) {
    double theta = lon1 - lon2;
    double degLat1 = deg2rad(lat1);
    double degLat2 = deg2rad(lat2);
    double dist = Math.sin(degLat1) * Math.sin(degLat2) +
        Math.cos(degLat1) * Math.cos(degLat2) * Math.cos(deg2rad(theta));
    dist = Math.acos(dist);
    dist = rad2deg(dist);
    return dist * 60 * 1.1515;
  }

  // assume kdtree node x is lat, and y is lng, return distance in miles between kdtree's max and min
  public static double kdtreeDistanceInMiles(KDTreeNode node) {
    return Distance.miles(node.getMaxX(), node.getMaxY(), node.getMinX(), node.getMinY());
  }

  public static double kilometers(double lat1, double lon1, double lat2, double lon2) {
    return miles(lat1, lon1, lat2, lon2) * 1.609344;
  }

  public static double nauticalMiles(double lat1, double lon1, double lat2, double lon2) {
    return miles(lat1, lon1, lat2, lon2) * 0.8684;
  }

  private static double deg2rad(double deg) {
    return (deg * Math.PI / 180.0);
  }

  private static double rad2deg(double rad) {
    return (rad * 180 / Math.PI);
  }
}
