package com.airbnb.aerosolve.core.util;

/**
 * Utilities for reinforcement learning
 */

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.models.AbstractModel;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

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
    double nextQ = 0.0f;
    if (nextStateAction != null) {
      nextQ = model.scoreItem(nextStateAction);
    }
   
    double currentQ = model.scoreItem(stateAction);
    double expectedQ = reward + discountRate * nextQ;
    double grad = currentQ - expectedQ;
    model.onlineUpdate(grad, learningRate, stateAction);
  }

  // Picks a random action with probability epsilon.
  public static int epsilonGreedyPolicy(AbstractModel model, ArrayList<FeatureVector> stateAction,
                                        double epsilon, Random rnd) {
    if (rnd.nextFloat() <= epsilon) {
      return rnd.nextInt(stateAction.size());
    }
    int bestAction = 0;
    double bestScore = model.scoreItem(stateAction.get(0));
    for (int i = 1; i < stateAction.size(); i++) {
      FeatureVector sa = stateAction.get(i);
      double score = model.scoreItem(sa);
      if (score > bestScore) {
        bestAction = i;
        bestScore = score;
      }
    }
    return bestAction;
  }

 // Uses softmax to determine the action.
 // As temperature approaches zero the action approaches the greedy action.
 public static int softmaxPolicy(AbstractModel model, ArrayList<FeatureVector> stateAction,
                                 double temperature, Random rnd) {
   int count = stateAction.size();
   double[] scores = new double[count];
   double[] cumScores = new double[count];
   double maxVal = -1e10d;
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
   double threshold = rnd.nextFloat() * cumScores[count - 1];
   for (int i = 0; i < count; i++) {
     if (threshold <= cumScores[i]) {
       return i;
     }
   }
   return 0;
 }
}