package com.airbnb.aerosolve.core.util;

/**
 * Utilities for reinforcement learning
 */

import com.airbnb.aerosolve.core.models.AbstractModel;
import com.airbnb.aerosolve.core.FeatureVector;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;

import org.apache.commons.codec.binary.Base64;
import org.apache.thrift.TDeserializer;
import org.apache.thrift.TSerializer;
import org.apache.thrift.TBase;

import java.io.Serializable;
import java.util.*;

public class ReinforcementLearning implements Serializable {
  // Updates a model using SARSA
  // https://en.wikipedia.org/wiki/State-Action-Reward-State-Action
  // If the nextState is terminal, pass in null
  public static void updateSARSA(AbstractModel model,
                                 FeatureVector stateAction,
                                 float reward,
                                 FeatureVector nextStateAction,
                                 float learningRate,
                                 float decay) {
    Map<String, Map<String, Double>> flatSA = Util.flattenFeature(stateAction);
    float nextQ = 0.0f;
    if (nextStateAction != null) {
      nextQ = model.scoreItem(nextStateAction);
    }
   
    float currentQ = model.scoreItem(stateAction);
    float expectedQ = reward + decay * nextQ;
    float grad = currentQ - expectedQ;
    model.onlineUpdate(grad, learningRate, flatSA);
  }

  // Picks a random action with probability epsilon.
  public static int epsilonGreedyPolicy(AbstractModel model, ArrayList<FeatureVector> stateAction, float epsilon, Random rnd) {
    if (rnd.nextFloat() <= epsilon) {
      return rnd.nextInt(stateAction.size());
    }
    int bestAction = 0;
    float bestScore = model.scoreItem(stateAction.get(0));
    for (int i = 1; i < stateAction.size(); i++) {
      FeatureVector sa = stateAction.get(i);
      float score = model.scoreItem(sa);
      if (score > bestScore) {
        bestAction = i;
        bestScore = score;
      }
    }
    return bestAction;
  }
}