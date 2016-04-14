package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.DictionaryEntry;
import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.TransformTestingHelper;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * @author Hector Yee
 */
@Slf4j
public class StringDictionaryTest {
  private final FeatureRegistry registry = new FeatureRegistry();

  StringDictionary makeDictionary() {
    StringDictionary dict = new StringDictionary();
    DictionaryEntry result = dict.getEntry("foo", "bar");
    assertEquals(null, result);
    int idx = dict.possiblyAdd(registry.feature("LOC", "lat"), 0.1, 0.1);
    assertEquals(0, idx);
    idx = dict.possiblyAdd(registry.feature("LOC", "lng"), 0.2, 0.2);
    assertEquals(1, idx);
    idx = dict.possiblyAdd(registry.feature("foo", "bar"), 0.3, 0.3);
    assertEquals(2, idx);
    return dict;
  }

   @Test
  public void testStringDictionaryAdd() {
     StringDictionary dict = makeDictionary();
     assertEquals(3, dict.getDictionary().getEntryCount());
     assertEquals(0, dict.getEntry("LOC", "lat").getIndex());
     assertEquals(1, dict.getEntry("LOC", "lng").getIndex());
     assertEquals(2, dict.getEntry("foo", "bar").getIndex());
     int result = dict.possiblyAdd(registry.feature("foo", "bar"), 0.0, 0.0);
     assertEquals(-1, result);
     assertEquals(3, dict.getDictionary().getEntryCount());     
  }
   
  @Test
  public void testStringDictionaryVector() {
    MultiFamilyVector vector = TransformTestingHelper.makeEmptyVector(registry);
    Family loc = registry.family("LOC");
    Family foo = registry.family("foo");

    vector.put(loc.feature("lat"), 1.0);
    vector.put(loc.feature("lng"), 2.0);
    vector.put(foo.feature("bar"), 3.0);
    vector.put(foo.feature("baz"), 4.0);
    
    StringDictionary dict = makeDictionary();

    FloatVector vec = dict.makeVectorFromSparseFloats(vector);
    assertEquals(3, vec.values.length);
    assertEquals(0.1 * (1.0 - 0.1), vec.values[0], 0.1f);
    assertEquals(0.2 * (2.0 - 0.2), vec.values[1], 0.1f);
    assertEquals(0.3 * (3.0 - 0.3), vec.values[2], 0.1f);
  }

 }
