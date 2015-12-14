package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.models.AbstractModel;
import com.airbnb.aerosolve.core.models.KernelModel;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class ReinforcementLearningTest {
  private static final Logger log = LoggerFactory.getLogger(ReinforcementLearningTest.class);
  
  StringDictionary makeDictionary() {
    StringDictionary dict = new StringDictionary();
    // The locations vary between 0 and 10
    dict.possiblyAdd("S", "x", 5.0, 0.2);
    dict.possiblyAdd("S", "y", 5.0, 0.2);
    // The actions are +/- 1
    dict.possiblyAdd("A", "dx", 0.0, 1.0);
    dict.possiblyAdd("A", "dy", 0.0, 1.0);
    return dict;
  }
  
  AbstractModel makeModel() {
    KernelModel model = new KernelModel();
    StringDictionary dict = makeDictionary(); 
    model.setDictionary(dict);
    List<SupportVector> supportVectors = model.getSupportVectors();
    Random rnd = new Random(12345);
    for (int i = 0; i < 100; i++) {
      FloatVector vec = new FloatVector(4);
      for (int j = 0; j < 4; j++) {
        vec.values[j] = 2.0f * rnd.nextFloat() - 1.0f;        
      }
      supportVectors.add(new SupportVector(vec, FunctionForm.RADIAL_BASIS_FUNCTION, 2.0f, 0.0f));
    }
    return model;
  }

  public FeatureVector makeFeatureVector(double x, double y, double dx, double dy) {
    HashMap floatFeatures = new HashMap<Double, HashMap<Double, String>>();
    HashMap stateFeatures = new HashMap<Double, String>();
    stateFeatures.put("x", x);
    stateFeatures.put("y", y);
    HashMap actionFeatures = new HashMap<Double, String>();
    actionFeatures.put("dx", dx);
    actionFeatures.put("dy", dy);
    floatFeatures.put("S", stateFeatures);
    floatFeatures.put("A", actionFeatures);
    FeatureVector featureVector = new FeatureVector();
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  @Test
  public void testSARSA() {
    AbstractModel model = makeModel();
    
  }
}