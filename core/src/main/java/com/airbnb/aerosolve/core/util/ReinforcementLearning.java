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
                                 float discountRate) {
    Map<String, Map<String, Double>> flatSA = Util.flattenFeature(stateAction);
    float nextQ = 0.0f;
    if (nextStateAction != null) {
      nextQ = model.scoreItem(nextStateAction);
    }
   
    float currentQ = model.scoreItem(stateAction);
    float expectedQ = reward + discountRate * nextQ;
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

 // Uses softmax to determine the action.
 // As temperature approaches zero the action approaches the greedy action.
 public static int softmaxPolicy(AbstractModel model, ArrayList<FeatureVector> stateAction, float temperature, Random rnd) {
   int count = stateAction.size();
   float[] scores = new float[count];
   float[] cumScores = new float[count];
   float maxVal = -1e10f;
   for (int i = 0; i < count; i++) {
     FeatureVector sa = stateAction.get(i);
     scores[i] = model.scoreItem(sa);
     maxVal = Math.max(maxVal, scores[i]);
   }
   for (int i = 0; i < count; i ++) {
     scores[i] = (float) Math.exp((scores[i] - maxVal) / temperature);
     cumScores[i] = scores[i];
     if (i > 0) {
       cumScores[i] += cumScores[i - 1];
     }
   }
   float threshold = rnd.nextFloat() * cumScores[count - 1];
   for (int i = 0; i < count; i++) {
     if (threshold <= cumScores[i]) {
       return i;
     }
   }
   return 0;
 }
}