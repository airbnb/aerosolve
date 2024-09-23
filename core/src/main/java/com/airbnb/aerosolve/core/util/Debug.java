package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import lombok.extern.slf4j.Slf4j;
import org.apache.thrift.TDeserializer;
import org.apache.thrift.TSerializer;
import org.apache.thrift.protocol.TBinaryProtocol;

import java.io.FileOutputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Set;

@Slf4j
public class Debug  {
  public static int printDiff(FeatureVector a, FeatureVector b) {
    final Map<String, Set<String>> stringFeaturesA = a.getStringFeatures();
    final Map<String, Set<String>> stringFeaturesB = b.getStringFeatures();
    int diff = printDiff(stringFeaturesA, stringFeaturesB);
    final Map<String, Map<String, Double>> floatFeaturesA = a.getFloatFeatures();
    final Map<String, Map<String, Double>> floatFeaturesB = b.getFloatFeatures();
    diff += printFloatDiff(floatFeaturesA, floatFeaturesB);
    return diff;
  }

  private static int printFloatDiff(Map<String, Map<String, Double>> a,
                                        Map<String, Map<String, Double>> b) {
    int diff = 0;
    for (Map.Entry<String, Map<String, Double>> entry : a.entrySet()) {
      String key = entry.getKey();
      Map<String, Double> bSet = b.get(key);
      if (bSet == null) {
        log.info("b miss float family {}", key);
        diff++;
      } else {
        diff += printMapDiff(entry.getValue(), bSet);
      }
    }

    for (Map.Entry<String, Map<String, Double>> entry : b.entrySet()) {
      String key = entry.getKey();
      Map<String, Double> bSet = a.get(key);
      if (bSet == null) {
        log.info("a miss float family {}", key);
        diff++;
      }
    }
    return diff;
  }

  private static int printMapDiff(Map<String, Double> a, Map<String, Double> b) {
    int diff = 0;
    for (Map.Entry<String, Double> entry : a.entrySet()) {
      String key = entry.getKey();
      Double bValue = b.get(key);
      if (bValue == null) {
        log.info("b miss feature {} {}", key, entry.getValue());
        diff++;
      } else {
        if (Math.abs(bValue- entry.getValue()) > 0.01) {
          log.info("feature {} a: {}, b: {}", key, entry.getValue(), bValue);
          diff++;
        }
      }
    }

    for (Map.Entry<String, Double> entry : b.entrySet()) {
      String key = entry.getKey();
      Double bValue = a.get(key);
      if (bValue == null) {
        log.info("a miss feature {} {}", key, entry.getValue());
        diff++;
      }
    }
    return diff;
  }


  public static int printDiff(Map<String, Set<String>> a, Map<String, Set<String>> b) {
    int diff = 0;
    for (Map.Entry<String, Set<String>> entry : a.entrySet()) {
      String key = entry.getKey();
      Set<String> bSet = b.get(key);
      if (bSet == null) {
        log.info("b miss string family {}", key);
        diff++;
      } else {
        diff += printDiff(entry.getValue(), bSet);
      }
    }

    for (Map.Entry<String, Set<String>> entry : b.entrySet()) {
      String key = entry.getKey();
      Set<String> bSet = a.get(key);
      if (bSet == null) {
        log.info("a miss string family {}", key);
        diff++;
      }
    }
    return diff;
  }

  private static int printDiff(Set<String> a, Set<String> b) {
    int diff = 0;
    for(String s : a) {
      if (!b.contains(s)) {
        log.info("b missing {}", s);
        diff++;
      }
    }
    for(String s : b) {
      if (!a.contains(s)) {
        log.info("a missing {}", s);
        diff++;
      }
    }
    return diff;
  }

  /*
  loadExampleFromResource read example from resources folder, i.e. test/resources
  use it on unit test to load example from disk
 */
  public static Example loadExampleFromResource(String name) {
    URL url = Debug.class.getResource("/" + name);
    try {
      Path path = Paths.get(url.toURI());
      byte[] bytes = Files.readAllBytes(path);
      TDeserializer deserializer = new TDeserializer(new TBinaryProtocol.Factory());
      Example example = new Example();
      deserializer.deserialize(example, bytes);
      return example;
    } catch (Exception e) {
      e.printStackTrace();
    }
    assert(false);
    return null;
  }

  // Save example to path
  // If you hit permission error, touch and chmod the file
  public static void saveExample(Example example, String path) {
    try {
      TSerializer serializer = new TSerializer(new TBinaryProtocol.Factory());
      byte[] buf = serializer.serialize(example);
      FileOutputStream fos = new FileOutputStream(path);
      fos.write(buf);
      fos.close();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
