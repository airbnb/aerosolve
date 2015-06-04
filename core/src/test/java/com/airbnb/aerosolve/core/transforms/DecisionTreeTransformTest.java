package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.models.DecisionTreeModel;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class DecisionTreeTransformTest {
  private static final Logger log = LoggerFactory.getLogger(DecisionTreeTransformTest.class);

  public FeatureVector makeFeatureVector(double x, double y) {
    Map<String, Set<String>> stringFeatures = new HashMap<>();
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Set list = new HashSet<String>();
    list.add("aaa");
    list.add("bbb");
    stringFeatures.put("strFeature1", list);

    Map<String, Double> map = new HashMap<>();
    map.put("x", x);
    map.put("y", y);
    floatFeatures.put("loc", map);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  public String makeConfig() {
    return "test_tree {\n" +
        " transform : decision_tree\n" +
        " output_leaves : \"LEAF\" \n" +
        " output_score_family : \"SCORE\" \n" +
        " output_score_name : \"TREE0\" \n" +
        " nodes : [\n" +
        "   \"P,0,loc,x,2.000000,1,2\" \n" +
        "   \"P,1,loc,y,0.000000,3,4\" \n" +
        "   \"P,2,loc,y,1.000000,5,6\" \n" +
        "   \"L,3,0.250000,BOTTOM_LEFT\" \n" +
        "   \"L,4,-0.500000,TOP_LEFT\" \n" +
        "   \"L,5,-0.750000,BOTTOM_RIGHT\" \n" +
        "   \"L,6,1.000000,TOP_RIGHT\" \n" +
        " ]\n" +
        "}";
  }

  /*
   * XOR like decision regions
   * 
   *         x = 2
   *          |
   *    -0.5  |   1.0
   *          |-------- y = 1
   *   -------- y = 0
   *          |
   *     0.25 |   -0.75
   *          |
   */

  public DecisionTreeModel makeTree() {
    ArrayList<ModelRecord> records = new ArrayList<>();
    DecisionTreeModel tree = new DecisionTreeModel();
    tree.setStumps(records);

    // 0 - an x split at 2
    ModelRecord record = new ModelRecord();
    record.setFeatureFamily("loc");
    record.setFeatureName("x");
    record.setThreshold(2.0);
    record.setLeftChild(1);
    record.setRightChild(2);
    records.add(record);

    // 1 - a y split at 0
    record = new ModelRecord();
    record.setFeatureFamily("loc");
    record.setFeatureName("y");
    record.setThreshold(0.0);
    record.setLeftChild(3);
    record.setRightChild(4);
    records.add(record);

    // 2 - a y split at 1
    record = new ModelRecord();
    record.setFeatureFamily("loc");
    record.setFeatureName("y");
    record.setThreshold(1.0);
    record.setLeftChild(5);
    record.setRightChild(6);
    records.add(record);

    // 3  a leaf
    record = new ModelRecord();
    record.setFeatureWeight(0.25);
    records.add(record);

    // 4  a leaf
    record = new ModelRecord();
    record.setFeatureWeight(-0.5);
    records.add(record);

    // 5  a leaf
    record = new ModelRecord();
    record.setFeatureWeight(-0.75);
    records.add(record);

    // 6  a leaf
    record = new ModelRecord();
    record.setFeatureWeight(1.0);
    records.add(record);

    return tree;
  }

  @Test
  public void testToHumanReadableConfig() {
    DecisionTreeModel tree = makeTree();
    String result = tree.toHumanReadableTransform();
    log.info(result);
    String tokens[] = result.split("\n");
    assertEquals(9, tokens.length);
    assertTrue(tokens[3].contains("P,2,loc,y,1.000000,5,6"));
    assertTrue(tokens[4].contains("L,3,0.250000,LEAF_3"));
  }

  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_tree");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);
    assertTrue(featureVector.getStringFeatures() == null);
  }

  public void testTransformAt(double x, double y, String expectedLeaf, double expectedOutput) {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_tree");

    FeatureVector featureVector;
    featureVector = makeFeatureVector(x, y);
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertEquals(2, stringFeatures.size());

    Set<String> out = featureVector.stringFeatures.get("LEAF");
    for (String entry : out) {
      log.info(entry);
    }
    assertTrue(out.contains(expectedLeaf));

    Map<String, Double> treeOutput = featureVector.floatFeatures.get("SCORE");
    assertTrue(treeOutput.containsKey("TREE0"));
    assertEquals(expectedOutput, treeOutput.get("TREE0"), 0.1);
  }

  @Test
  public void testTransform() {
    testTransformAt(10.0, 10.0, "TOP_RIGHT", 1.0);
    testTransformAt(10.0, -10.0, "BOTTOM_RIGHT", -0.75);
    testTransformAt(-10.0, 10.0, "TOP_LEFT", -0.5);
    testTransformAt(-10.0, -10.0, "BOTTOM_LEFT", 0.25);
  }
}