package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
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
    int result = dict.getIndex("foo", "bar");
    assertEquals(-1, result);
    result = dict.possiblyAdd("LOC", "lat");
    assertEquals(0, result);
    result = dict.possiblyAdd("LOC", "lng");
    assertEquals(1, result);
    result = dict.possiblyAdd("foo", "bar");
    assertEquals(2, result);
    return dict;
  }

   @Test
  public void testStringDictionaryAdd() {
     StringDictionary dict = makeDictionary();
     assertEquals(3, dict.getEntryCount());
     assertEquals(0, dict.getIndex("LOC", "lat"));
     assertEquals(1, dict.getIndex("LOC", "lng"));
     assertEquals(2, dict.getIndex("foo", "bar"));
     int result = dict.possiblyAdd("foo", "bar");
     assertEquals(-1, result);
     assertEquals(3, dict.getEntryCount());     
  }

 }
