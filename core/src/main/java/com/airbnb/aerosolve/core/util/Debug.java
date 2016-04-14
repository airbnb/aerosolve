package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.SimpleExample;
import com.google.common.collect.MapDifference;
import com.google.common.collect.Maps;
import java.io.FileOutputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import lombok.extern.slf4j.Slf4j;
import org.apache.thrift.TDeserializer;
import org.apache.thrift.TSerializer;
import org.apache.thrift.protocol.TBinaryProtocol;

@Slf4j
public class Debug  {
  public static int printDiff(FeatureVector a, FeatureVector b) {
    MapDifference<Feature, Double> diff = Maps.difference(a, b);

    // TODO (Brad): We can format this differently or something if needed.
    // It should print out pretty reasonably though.
    log.info(diff.toString());
    return diff.entriesDiffering().size() +
           diff.entriesOnlyOnLeft().size() +
           diff.entriesOnlyOnRight().size();
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
      FeatureRegistry registry = new FeatureRegistry();
      Example example = new SimpleExample(registry);

      // TODO (Brad): BEFORE MERGE!! Fix up Thrift stuff.
      //deserializer.deserialize(example, bytes);
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
    TSerializer serializer = new TSerializer(new TBinaryProtocol.Factory());
    try {
      // TODO (Brad): BEFORE MERGE!! Fix up Thrift stuff.
      //byte[] buf = serializer.serialize(example);
      FileOutputStream fos = new FileOutputStream(path);
      //fos.write(buf);
      fos.close();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
