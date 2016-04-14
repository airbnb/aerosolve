package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.SimpleExample;
import com.google.common.collect.MapDifference;
import com.google.common.collect.Maps;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.codec.binary.Base64;
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
  public static Example loadExampleFromResource(String name, FeatureRegistry registry) {
    URL url = Debug.class.getResource("/" + name);
    try {
      Path path = Paths.get(url.toURI());
      byte[] bytes = Files.readAllBytes(path);
      return Util.decodeExample(new String(Base64.encodeBase64(bytes)), registry);
    } catch (Exception e) {
      e.printStackTrace();
    }
    assert(false);
    return null;
  }

  // Save example to path
  // If you hit permission error, touch and chmod the file
  public static void saveExample(Example example, String path) {
    // TODO (Brad): This base64 encoding stuff is crazy.  Let's fix that.
    String encoded = Util.encodeExample(example);
    byte[] bytes = Base64.decodeBase64(encoded);
    try {
      FileOutputStream out = new FileOutputStream(path);
      out.write(bytes);
      out.close();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
