package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.models.AbstractModel;
import com.airbnb.aerosolve.core.models.KernelModel;
import com.airbnb.aerosolve.core.perf.Family;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.TransformTestingHelper;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
@Slf4j
public class ReinforcementLearningTest {
  private final FeatureRegistry registry = new FeatureRegistry();

  StringDictionary makeDictionary() {
    StringDictionary dict = new StringDictionary();
    // The locations vary between 0 and 10
    dict.possiblyAdd("S", "x", 0.0, 1.0);
    dict.possiblyAdd("S", "y", 0.0, 1.0);
    // The actions are +/- 1
    dict.possiblyAdd("A", "dx", 0.0, 1.0);
    dict.possiblyAdd("A", "dy", 0.0, 1.0);
    return dict;
  }
  
  public Map<String, Double> makeState(double x, double y) {
    Map<String, Double> stateFeatures = new HashMap<>();
    stateFeatures.put("x", x);
    stateFeatures.put("y", y);
    return stateFeatures;
  }
  
  public Map<String, Double> makeAction(double dx, double dy) {
    Map<String, Double> actionFeatures = new HashMap<>();
    actionFeatures.put("dx", dx);
    actionFeatures.put("dy", dy);
    return actionFeatures;
  }

  public MultiFamilyVector makeFeatureVector(Map<String, Double> state,
                                             Map<String, Double> action) {
    MultiFamilyVector vector = TransformTestingHelper.makeEmptyVector(registry);
    loadVector(vector, registry.family("S"), state);
    loadVector(vector, registry.family("A"), action);

    return vector;
  }

  private void loadVector(MultiFamilyVector vector, Family family, Map<String, Double> values) {
    for (Map.Entry<String, Double> entry : values.entrySet()) {
      vector.put(family.feature(entry.getKey()), entry.getValue());
    }
  }

  public ArrayList<FeatureVector> stateActions(double x, double y) {
    Map<String, Double> currState = makeState(x, y);
    ArrayList<FeatureVector> potential = new ArrayList<>();
    Map<String, Double> up = makeAction(0.0f, 1.0f);
    Map<String, Double> down = makeAction(0.0f, -1.0f);
    Map<String, Double> left = makeAction(-1.0f, 0.0f);
    Map<String, Double> right = makeAction(1.0f, 0.0f);

    potential.add(makeFeatureVector(currState, up));
    potential.add(makeFeatureVector(currState, down));
    potential.add(makeFeatureVector(currState, left));
    potential.add(makeFeatureVector(currState, right));
    return potential;
  }

  AbstractModel makeModel() {
    KernelModel model = new KernelModel(registry);
    StringDictionary dict = makeDictionary();
    model.setDictionary(dict);
    List<SupportVector> supportVectors = model.getSupportVectors();
    Random rnd = new Random(12345);
    for (double x = 0.0; x <= 10.0; x += 1.0) {
      for (double y = 0.0; y <= 10.0; y += 1.0) {
        ArrayList<FeatureVector> potential = stateActions(x, y);
        for (FeatureVector sa : potential) {
          FloatVector vec = dict.makeVectorFromSparseFloats(sa);
          SupportVector sv = new SupportVector(vec, FunctionForm.RADIAL_BASIS_FUNCTION, 1.0f, 0.0f);
          supportVectors.add(sv);
        }
      }
    }
    return model;
  }

  // In the cliffworld scenario, the agent starts at (0,0)
  // There is a cliff in the region (2.0, 0.0) to (8.0, 2.0) if the agent steps in the cliff it gets a reward of -100 and teleports back to the start.
  // Otherwise the reward is -1. If it hits the goal the reward is 10.0
  //
  //
  //
  //
   //  Start    CLIFF    Goal
  //   (0,0)    CLIFF    (10, 0)
  public float runEpisode(AbstractModel model, int epoch, Random rnd, boolean epsGreedy) {
    float sum = 0.0f;
    double x = 0.0;
    double y = 0.0;
    FeatureVector prevStateAction = null;
    float prevReward = 0.0f;
    FeatureVector currStateAction = null;
    float epsilon = 0.1f;
    float learningRate = 0.1f;
    float temperature = 10.0f;
    if (epoch > 90) {
      epsilon = 0.00f;
      temperature = 0.1f;
      learningRate = 0.001f;
    }
    boolean done = false;
    while (!done) {
      ArrayList<FeatureVector> potential = stateActions(x, y);

      int pick = 0;
      if (epsGreedy) {
        pick = ReinforcementLearning.epsilonGreedyPolicy(model, potential, epsilon, rnd);
      } else {
        pick = ReinforcementLearning.softmaxPolicy(model, potential, temperature, rnd);
      }
      currStateAction = potential.get(pick);
      switch (pick) {
        case 0: y = y + 1; break;
        case 1: y = y - 1; break;
        case 2: x = x - 1; break;
        case 3: x = x + 1;break;
      }

      float reward = -1.0f;
      // Boundaries
      if (x < 0.0) {
        reward = -10.f;
        x = 0.0;
      }
      if (x > 10.0) {
        reward = -10.f;
        x = 10.0;
      }
      if (y < 0.0) {
        reward = -10.f;
        y = 0.0;
      }
      if (y > 10.0) {
        reward = -10.f;
        y = 10.0;
      }
      // The cliff teleports back to the start.
      if (x >= 2.0 && x <= 8.0 && y <= 2.0) {
        x = y = 0.0;
        reward = -100.0f;
      }
      // Goal
      if (x >= 9.0f && y <= 1.0f) {
        reward = 10.0f;
        log.info("GOAL");
        done = true;
      }
      if (prevStateAction != null) {
        if (done) {
          ReinforcementLearning.updateSARSA(model, currStateAction, reward, null, learningRate, 1.0f);
        } else {
          ReinforcementLearning.updateSARSA(model, prevStateAction, prevReward, currStateAction, learningRate, 1.0f);
        }
      }
      prevStateAction = currStateAction;
      sum += reward;
      prevReward = reward;
    }
    return sum;
  }

  @Test
  public void testSARSAGreedy() {
    AbstractModel model = makeModel();
    Random rnd = new Random(1234);
    float sum = 0.0f;
    for (int i = 0; i < 120; i++) {
       sum = runEpisode(model, i, rnd, true);
       log.info("Episode " + i + " score " + sum);
    }
    assertTrue(sum > -10.0f);
  }

  @Test
  public void testSARSASoftmax() {
    AbstractModel model = makeModel();
    Random rnd = new Random(1234);
    float sum = 0.0f;
    for (int i = 0; i < 120; i++) {
       sum = runEpisode(model, i, rnd, false);
       log.info("Episode " + i + " score " + sum);
    }
    assertTrue(sum > -10.0f);
  }

}