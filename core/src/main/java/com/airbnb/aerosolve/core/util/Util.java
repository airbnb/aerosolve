package com.airbnb.aerosolve.core.util;

/**
 * @author hector_yee
 *
 * Utilities for machine learning
 */

import com.airbnb.aerosolve.core.*;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import org.apache.commons.codec.binary.Base64;
import org.apache.thrift.TBase;
import org.apache.thrift.TDeserializer;
import org.apache.thrift.TSerializer;

import java.io.*;
import java.util.*;
import java.util.zip.GZIPInputStream;

public class Util implements Serializable {
  private static double LOG2 = Math.log(2);
  // Coder / decoder utilities for various protos. This makes it easy to
  // manipulate in spark. e.g. if we  wanted to see the 50 weights in a model
  // val top50 = sc.textFile("model.bz2").map(Util.decodeModel).sortBy(x => -x.weight).take(50);
  public static String encode(TBase obj) {
    TSerializer serializer = new TSerializer();
    try {
      byte[] bytes = serializer.serialize(obj);
      return new String(Base64.encodeBase64(bytes));
    } catch (Exception e) {
      return "";
    }
  }
  public static FeatureVector decodeFeatureVector(String str) {
    return decode(FeatureVector.class, str);
  }

  public static FeatureVector createNewFeatureVector() {
    FeatureVector featureVector = new FeatureVector();
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();
    Map<String, Set<String>> stringFeatures = new HashMap<>();
    featureVector.setFloatFeatures(floatFeatures);
    featureVector.setStringFeatures(stringFeatures);

    return featureVector;
  }

  public static Example createNewExample() {
    Example example = new Example();
    example.setContext(createNewFeatureVector());
    example.setExample(new ArrayList<FeatureVector>());

    return example;
  }

  public static Example decodeExample(String str) {
    return decode(Example.class, str);
  }

  public static ModelRecord decodeModel(String str) {
    return decode(ModelRecord.class, str);
  }

  public static <T extends TBase> T decode(T base, String str) {
    try {
      byte[] bytes = Base64.decodeBase64(str.getBytes());
      TDeserializer deserializer = new TDeserializer();
      deserializer.deserialize(base, bytes);
    } catch (Exception e) {
      e.printStackTrace();
    }
    return base;
  }

  public static <T extends TBase> T decode(Class<T> clazz, String str) {
    try {
      return decode(clazz.newInstance(), str);
    } catch (InstantiationException e) {
      e.printStackTrace();
    } catch (IllegalAccessException e) {
      e.printStackTrace();
    }
    return null;
  }

  public static KDTreeNode decodeKDTreeNode(String str) {
    return decode(KDTreeNode.class, str);
  }

  public static <T extends TBase> List<T> readFromGzippedStream(Class<T> clazz, InputStream inputStream) {
    try {
      if (inputStream != null) {
        GZIPInputStream gzipInputStream = new GZIPInputStream(inputStream);
        BufferedReader reader = new BufferedReader(new InputStreamReader(gzipInputStream));
        List<T> list = new ArrayList<>();
        String line = reader.readLine();
        while(line != null) {
          T t = Util.decode(clazz, line);
          if (t == null) {
            assert (false);
            return Collections.EMPTY_LIST;
          }
          list.add(t);
          line = reader.readLine();
        }
        return list;
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
    return Collections.EMPTY_LIST;
  }

  public static void optionallyCreateStringFeatures(FeatureVector featureVector) {
    if (featureVector.getStringFeatures() == null) {
      Map<String, Set<String>> stringFeatures = new HashMap<>();
      featureVector.setStringFeatures(stringFeatures);
    }
  }

  public static void optionallyCreateFloatFeatures(FeatureVector featureVector) {
    if (featureVector.getFloatFeatures() == null) {
      Map<String, Map<String, Double>> floatFeatures = new HashMap<>();
      featureVector.setFloatFeatures(floatFeatures);
    }
  }

  public static void setStringFeature(
      FeatureVector featureVector,
      String family,
      String value) {
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    if (stringFeatures == null) {
      stringFeatures = new HashMap<>();
      featureVector.setStringFeatures(stringFeatures);
    }
    Set<String> stringFamily = getOrCreateStringFeature(family, stringFeatures);
    stringFamily.add(value);
  }

  public static Set<String> getOrCreateStringFeature(
      String name,
      Map<String, Set<String>> stringFeatures) {
    Set<String> output = stringFeatures.get(name);
    if (output == null) {
      output = new HashSet<>();
      stringFeatures.put(name, output);
    }
    return output;
  }

  public static Map<String, Double> getOrCreateFloatFeature(
      String name,
      Map<String, Map<String, Double>> floatFeatures) {
    Map<String, Double> output = floatFeatures.get(name);
    if (output == null) {
      output = new HashMap<>();
      floatFeatures.put(name, output);
    }
    return output;
  }

  public static Map<String, List<Double>> getOrCreateDenseFeatures(FeatureVector featureVector) {
    if (featureVector.getDenseFeatures() == null) {
      Map<String, List<Double>> dense = new HashMap<>();
      featureVector.setDenseFeatures(dense);
    }
    return featureVector.getDenseFeatures();
  }

  public static void setDenseFeature(
      FeatureVector featureVector,
      String name,
      List<Double> value) {
    Map<String, List<Double>> denseFeatures = featureVector.getDenseFeatures();
    if (denseFeatures == null) {
      denseFeatures = new HashMap<>();
      featureVector.setDenseFeatures(denseFeatures);
    }
    denseFeatures.put(name, value);
  }


  public static HashCode getHashCode(String family, String value) {
    Hasher hasher = Hashing.murmur3_128().newHasher();
    hasher.putBytes(family.getBytes());
    hasher.putBytes(value.getBytes());
    return hasher.hash();
  }

  public static <K, V extends Comparable<V>> Map.Entry<K, V>[] sortByValuesDesc(final Map<K, V> map) {
    if (map == null || map.size() == 0) {
      return null;
    }
    try {
      @SuppressWarnings("unchecked")
      Map.Entry<K, V>[] array = map.entrySet().toArray(new Map.Entry[map.size()]);

      Arrays.sort(array, new Comparator<Map.Entry<K, V>>() {
        public int compare(Map.Entry<K, V> e1, Map.Entry<K, V> e2) {
          return e2.getValue().compareTo(e1.getValue());
        }
      });

      return array;
    } catch (Exception e) {
    }
    return null;
  }

  public static <K extends Comparable<K>, V> Map.Entry<K, V>[] sortByKeysAsc(final Map<K, V> map) {
    if (map == null || map.size() == 0) {
      return null;
    }
    try {
      @SuppressWarnings("unchecked")
      Map.Entry<K, V>[] array = map.entrySet().toArray(new Map.Entry[map.size()]);

      Arrays.sort(array, new Comparator<Map.Entry<K, V>>() {
        public int compare(Map.Entry<K, V> e1, Map.Entry<K, V> e2) {
          return e1.getKey().compareTo(e1.getKey());
        }
      });

      return array;
    } catch (Exception e) {
    }
    return null;
  }

  public static double logBase2(double value) {
    return Math.log(value)/LOG2;
  }

  public static Map<Integer, Double> prepareRankMap(List<Double> scores, List<Double> utilities) {
    Map<Integer, Double> rankMap = new HashMap<Integer, Double>();
    if (scores == null || utilities == null || scores.size() != utilities.size()) {
      return rankMap;
    }
    Map<Integer, Double> scoresMap = new HashMap<>();
    for (int i = 0; i < scores.size(); i++) {
      scoresMap.put(i, scores.get(i));
    }
    Map.Entry<Integer, Double>[] kvs = sortByValuesDesc(scoresMap);
    for (int j = 0; j < kvs.length; j ++) {
      double util = utilities.get(kvs[j].getKey());
      rankMap.put(j, util);
    }
    return rankMap;
  }

  public static  Map<String, Map<String, Double>> flattenFeature(
      FeatureVector featureVector) {
    Map<String, Map<String, Double>> flatFeature = new HashMap<>();
    if (featureVector.stringFeatures != null) {
      for (Map.Entry<String, Set<String>> entry : featureVector.stringFeatures.entrySet()) {
        Map<String, Double> out = new HashMap<>();
        flatFeature.put(entry.getKey(), out);
        for (String feature : entry.getValue()) {
          out.put(feature, 1.0);
        }
      }
    }
    if (featureVector.floatFeatures != null) {
      for (Map.Entry<String, Map<String, Double>> entry : featureVector.floatFeatures.entrySet()) {
        Map<String, Double> out = new HashMap<>();
        flatFeature.put(entry.getKey(), out);
        for (Map.Entry<String, Double> feature : entry.getValue().entrySet()) {
          out.put(feature.getKey(), feature.getValue());
        }
      }
    }
    return flatFeature;
  }

  public static  Map<String, Map<String, Double>> flattenFeatureWithDropout(
      FeatureVector featureVector,
      double dropout) {
    Map<String, Map<String, Double>> flatFeature = new HashMap<>();
    if (featureVector.stringFeatures != null) {
      for (Map.Entry<String, Set<String>> entry : featureVector.stringFeatures.entrySet()) {
        Map<String, Double> out = new HashMap<>();
        flatFeature.put(entry.getKey(), out);
        for (String feature : entry.getValue()) {
          if (Math.random() < dropout) continue;
          out.put(feature, 1.0);
        }
      }
    }
    if (featureVector.floatFeatures != null) {
      for (Map.Entry<String, Map<String, Double>> entry : featureVector.floatFeatures.entrySet()) {
        Map<String, Double> out = new HashMap<>();
        flatFeature.put(entry.getKey(), out);
        for (Map.Entry<String, Double> feature : entry.getValue().entrySet()) {
          if (Math.random() < dropout) continue;
          out.put(feature.getKey(), feature.getValue());
        }
      }
    }
    return flatFeature;
  }

  public static class DebugDiffRecordComparator implements Comparator<DebugScoreDiffRecord> {
    @Override
    public int compare(DebugScoreDiffRecord e1, DebugScoreDiffRecord e2) {
      double v1 = Math.abs(e1.getFeatureWeightDiff());
      double v2 = Math.abs(e2.getFeatureWeightDiff());
      if (v1 > v2) {
        return -1;
      } else if (v1 < v2) {
        return 1;
      }
      return 0;
    }
  }

  private static Map<String, Map<String, Double>> debugScoreRecordListToMap(List<DebugScoreRecord> recordList){
    Map<String, Map<String, Double>> recordMap = new HashMap<>();

    for(int i = 0; i < recordList.size(); i++){
      String key = recordList.get(i).featureFamily + '\t' + recordList.get(i).featureName;
      Map<String, Double> record = new HashMap<>();
      record.put("featureValue", recordList.get(i).featureValue);
      record.put("featureWeight", recordList.get(i).featureWeight);
      recordMap.put(key, record);
    }
    return recordMap;
  }

  

  public static List<DebugScoreDiffRecord> compareDebugRecords(List<DebugScoreRecord> record1,
                                                               List<DebugScoreRecord> record2){
    List<DebugScoreDiffRecord> debugDiffRecord = new ArrayList<>();
    final String featureValue = "featureValue";
    final String featureWeight = "featureWeight";
    Set<String> keys = new HashSet();

    Map<String, Map<String, Double>> recordMap1 = debugScoreRecordListToMap(record1);
    Map<String, Map<String, Double>> recordMap2 = debugScoreRecordListToMap(record2);
    keys.addAll(recordMap1.keySet());
    keys.addAll(recordMap2.keySet());

    for(String key: keys){
      DebugScoreDiffRecord diffRecord = new DebugScoreDiffRecord();
      double fv1 = 0.0;
      double fv2 = 0.0;
      double fw1 = 0.0;
      double fw2 = 0.0;
      if(recordMap1.get(key) != null){
        fv1 = recordMap1.get(key).get(featureValue);
        fw1 = recordMap1.get(key).get(featureWeight);
      }
      if(recordMap2.get(key) != null){
        fv2 = recordMap2.get(key).get(featureValue);
        fw2 = recordMap2.get(key).get(featureWeight);
      }

      String[] fvString = key.split("\t");

      diffRecord.setFeatureFamily(fvString[0]);
      diffRecord.setFeatureName(fvString[1]);
      diffRecord.setFeatureValue1(fv1);
      diffRecord.setFeatureValue2(fv2);
      diffRecord.setFeatureWeight1(fw1);
      diffRecord.setFeatureWeight2(fw2);
      diffRecord.setFeatureWeightDiff(fw1 - fw2);
      debugDiffRecord.add(diffRecord);
    }
    Collections.sort(debugDiffRecord, new DebugDiffRecordComparator());
    return debugDiffRecord;
  }
}