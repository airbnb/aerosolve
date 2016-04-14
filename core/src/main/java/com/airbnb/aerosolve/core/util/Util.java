package com.airbnb.aerosolve.core.util;

/**
 * @author hector_yee
 *
 * Utilities for machine learning
 */

import com.airbnb.aerosolve.core.DebugScoreDiffRecord;
import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.KDTreeNode;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.ThriftExample;
import com.airbnb.aerosolve.core.ThriftFeatureVector;
import com.airbnb.aerosolve.core.perf.DenseVector;
import com.airbnb.aerosolve.core.perf.FamilyVector;
import com.airbnb.aerosolve.core.perf.FastMultiFamilyVector;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.FeatureValue;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.perf.SimpleExample;
import com.google.common.collect.Lists;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import com.google.common.primitives.Doubles;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.codec.binary.Base64;
import org.apache.thrift.TBase;
import org.apache.thrift.TDeserializer;
import org.apache.thrift.TSerializer;
import java.util.zip.GZIPInputStream;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.thrift.TBase;
import org.apache.thrift.TDeserializer;
import org.apache.thrift.TSerializer;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

@Slf4j
public class Util implements Serializable {

  // SimpleDateFormat is not thread safe. . .
  public static final DateTimeFormatter DATE_FORMAT = DateTimeFormat.forPattern("yyyy-MM-dd");

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

  public static String encodeFeatureVector(MultiFamilyVector vector) {
    return encode(getThriftFeatureVector(vector));
  }

  private static ThriftFeatureVector getThriftFeatureVector(MultiFamilyVector vector) {
    ThriftFeatureVector tVec = new ThriftFeatureVector();
    Map<String, Set<String>> stringFeatures = new HashMap<>();
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();
    Map<String, List<Double>> denseFeatures = new HashMap<>();
    for (FamilyVector familyVector : vector.families()) {
      String familyName = familyVector.family().name();
      // This might seem to break compatibility because we write all SparseVectors
      // as floatFeatures.  But it's fine as long as we don't try to read this vector in
      // old versions of aerosolve.
      if (familyVector instanceof DenseVector) {
        denseFeatures.put(familyName, Doubles.asList(((DenseVector) familyVector).getValues()));
      } else {
        floatFeatures.put(familyName, familyVector.entrySet()
                              .stream()
                              .map(e -> Pair.of(e.getKey().name(), e.getValue()))
                              .collect(Collectors.toMap(Pair::getKey, Pair::getValue)));
      }
    }
    tVec.setDenseFeatures(denseFeatures);
    tVec.setStringFeatures(stringFeatures);
    tVec.setFloatFeatures(floatFeatures);
    return tVec;
  }

  public static String encodeExample(Example example) {
    return encode(getThriftExample(example));
  }

  public static ThriftExample getThriftExample(Example example) {
    ThriftExample thriftExample = new ThriftExample();
    thriftExample.setContext(getThriftFeatureVector(example.context()));
    for (MultiFamilyVector innerVec : example) {
      thriftExample.addToExample(getThriftFeatureVector(innerVec));
    }
    return thriftExample;
  }

  public static MultiFamilyVector decodeFeatureVector(String str, FeatureRegistry registry) {
    ThriftFeatureVector tmp = new ThriftFeatureVector();
    try {
      byte[] bytes = Base64.decodeBase64(str.getBytes());
      TDeserializer deserializer = new TDeserializer();
      deserializer.deserialize(tmp, bytes);
    } catch (Exception e) {
      log.error("Error deserializing ThriftFeatureVector", e);
    }
    return new FastMultiFamilyVector(tmp, registry);
  }

  public static Example decodeExample(String str, FeatureRegistry registry) {
    ThriftExample tmp = new ThriftExample();
    try {
      byte[] bytes = Base64.decodeBase64(str.getBytes());
      TDeserializer deserializer = new TDeserializer();
      deserializer.deserialize(tmp, bytes);
    } catch (Exception e) {
      log.error("Error deserializing ThriftExample", e);
    }
    return new SimpleExample(tmp, registry);
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

  public static double euclideanDistance(double[] x, List<Float> y) {
    assert (x.length == y.size());
    double sum = 0;
    for (int i = 0; i < x.length; i++) {
      final double dp = x[i] - y.get(i);
      sum += dp * dp;
    }
    return Math.sqrt(sum);
  }

  public static double euclideanDistance(List<Double> x, List<Float> y) {
    assert (x.size() == y.size());
    double sum = 0;
    for (int i = 0; i < y.size(); i++) {
      final double dp = x.get(i) - y.get(i);
      sum += dp * dp;
    }
    return Math.sqrt(sum);
  }
}