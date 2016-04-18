package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.Set;

import static org.junit.Assert.*;

public class CutFloatTransformTest {
    private static final Logger log = LoggerFactory.getLogger(CapFloatTransformTest.class);

    public String makeConfig() {
        return "test_cut {\n" +
                " transform : cut_float\n" +
                " field1 : loc\n" +
                " upper_bound : 39.0\n" +
                " keys : [lat,long,z,aaa]\n" +
                "}";
    }

    public String makeConfigWithOutput() {
        return "test_cut {\n" +
                " transform : cut_float\n" +
                " field1 : loc\n" +
                " lower_bound : 1.0\n" +
                " upper_bound : 39.0\n" +
                " keys : [lat,long,z,aaa]\n" +
                " output : new_output \n" +
                "}";
    }

    public String makeConfigWithLowerBoundOnly() {
        return "test_cut {\n" +
                " transform : cut_float\n" +
                " field1 : loc\n" +
                " lower_bound : 1.0\n" +
                " keys : [lat,long,z,aaa]\n" +
                "}";
    }

    @Test
    public void testEmptyFeatureVector() {
        Config config = ConfigFactory.parseString(makeConfig());
        Transform transform = TransformFactory.createTransform(config, "test_cut");
        FeatureVector featureVector = new FeatureVector();
        transform.doTransform(featureVector);
        assertTrue(featureVector.getStringFeatures() == null);
    }

    @Test
    public void testTransform() {
        Config config = ConfigFactory.parseString(makeConfig());
        Transform transform = TransformFactory.createTransform(config, "test_cut");
        FeatureVector featureVector = TransformTestingHelper.makeFeatureVector();
        transform.doTransform(featureVector);
        Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
        assertTrue(stringFeatures.size() == 1);

        Map<String, Double> feat1 = featureVector.getFloatFeatures().get("loc");

        assertEquals(2, feat1.size());
        assertEquals(37.7, feat1.get("lat"), 0.1);
        assertNull(feat1.get("long"));
        assertEquals(-20.0, feat1.get("z"), 0.1);
    }

    @Test
    public void testTransformWithNewOutput() {
        Config config = ConfigFactory.parseString(makeConfigWithOutput());
        Transform transform = TransformFactory.createTransform(config, "test_cut");
        FeatureVector featureVector = TransformTestingHelper.makeFeatureVector();
        transform.doTransform(featureVector);
        Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
        assertTrue(stringFeatures.size() == 1);
        // original feature should not change
        Map<String, Double> feat1 = featureVector.getFloatFeatures().get("loc");
        assertEquals(3, feat1.size());
        assertEquals(37.7, feat1.get("lat"), 0.1);
        assertEquals(40.0, feat1.get("long"), 0.1);
        assertEquals(-20, feat1.get("z"), 0.1);

        // capped features are in a new feature family
        assertTrue(featureVector.getFloatFeatures().containsKey("new_output"));
        Map<String, Double> feat2 = featureVector.getFloatFeatures().get("new_output");
        assertEquals(1, feat2.size());
        assertEquals(37.7, feat2.get("lat"), 0.1);
        assertNull(feat2.get("long"));
        assertNull(feat2.get("z"));
    }

    @Test
    public void testTransformLowerBoundOnly() {
        Config config = ConfigFactory.parseString(makeConfigWithLowerBoundOnly());
        Transform transform = TransformFactory.createTransform(config, "test_cut");
        FeatureVector featureVector = TransformTestingHelper.makeFeatureVector();
        transform.doTransform(featureVector);
        Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
        assertTrue(stringFeatures.size() == 1);

        Map<String, Double> feat1 = featureVector.getFloatFeatures().get("loc");

        assertEquals(2, feat1.size());
        assertEquals(37.7, feat1.get("lat"), 0.1);
        assertEquals(40.0, feat1.get("long"), 0.1);
        assertNull(feat1.get("z"));
    }

}