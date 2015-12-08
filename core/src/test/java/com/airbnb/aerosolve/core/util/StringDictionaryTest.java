package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.Example;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.DictionaryEntry;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class StringDictionaryTest {
  private static final Logger log = LoggerFactory.getLogger(StringDictionaryTest.class);
  
  StringDictionary makeDictionary() {
    StringDictionary dict = new StringDictionary();
    DictionaryEntry result = dict.getEntry("foo", "bar");
    assertEquals(null, result);
    int idx = dict.possiblyAdd("LOC", "lat", 0.1, 0.1);
    assertEquals(0, idx);
    idx = dict.possiblyAdd("LOC", "lng", 0.2, 0.2);
    assertEquals(1, idx);
    idx = dict.possiblyAdd("foo", "bar", 0.3, 0.3);
    assertEquals(2, idx);
    return dict;
  }

   @Test
  public void testStringDictionaryAdd() {
     StringDictionary dict = makeDictionary();
     assertEquals(3, dict.getDictionary().getEntryCount());
     assertEquals(0, dict.getEntry("LOC", "lat").index);
     assertEquals(1, dict.getEntry("LOC", "lng").index);
     assertEquals(2, dict.getEntry("foo", "bar").index);
     int result = dict.possiblyAdd("foo", "bar", 0.0, 0.0);
     assertEquals(-1, result);
     assertEquals(3, dict.getDictionary().getEntryCount());     
  }
   
  @Test
  public void testStringDictionaryVector() {
    Map<String, Map<String, Double>> feature = new HashMap<>();
    Map<String, Double> loc = new HashMap<>();
    Map<String, Double> foo = new HashMap<>();

    feature.put("LOC", loc);
    feature.put("foo", foo);
    
    loc.put("lat", 1.0);
    loc.put("lng", 2.0);
    foo.put("bar", 3.0);
    foo.put("baz", 4.0);
    
    StringDictionary dict = makeDictionary();

    FloatVector vec = dict.makeVectorFromSparseFloats(feature);
    assertEquals(3, vec.values.length);
    assertEquals(0.1 * (1.0 - 0.1), vec.values[0], 0.1f);
    assertEquals(0.2 * (2.0 - 0.2), vec.values[1], 0.1f);
    assertEquals(0.3 * (3.0 - 0.3), vec.values[2], 0.1f);
  }

 }
