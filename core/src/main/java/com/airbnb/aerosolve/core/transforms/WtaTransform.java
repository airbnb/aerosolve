package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.DenseVector;
import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.FamilyVector;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.ConfigurableTransform;
import com.google.common.base.Preconditions;
import com.typesafe.config.Config;

import it.unimi.dsi.fastutil.ints.IntArrayFIFOQueue;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;
import org.hibernate.validator.constraints.NotEmpty;

import javax.validation.constraints.Max;
import javax.validation.constraints.NotNull;

/**
 * A transform that applies the winner takes all hash to
 * A set of dense feature families and emits tokens to a string feature family.
 * See "The power of comparative reasoning"
 * http://research.google.com/pubs/pub37298.html
 * For ease of use we use the recommended window size of 4 features
 * to generate 2-bit tokens
 * and pack each word with num_tokens_per_word of these.
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class WtaTransform extends ConfigurableTransform<WtaTransform> {
  private static final byte WINDOW_SIZE = 4;

  @NotNull
  @NotEmpty
  private List<String> familyNames;
  @NotNull
  private String outputFamilyName;
  // The seed of the random number generator.
  private int seed;
  // The number of words per feature.
  private int numWordsPerFeature;
  // The number of tokens per word.
  @Max(16)
  private int numTokensPerWord;

  @Setter(AccessLevel.NONE)
  private Random rnd;
  @Setter(AccessLevel.NONE)
  private List<Family> families;
  @Setter(AccessLevel.NONE)
  private Family outputFamily;

  @Override
  public WtaTransform configure(Config config, String key) {
    return
        familyNames(stringListFromConfig(config, key, ".field_names", true))
            .outputFamilyName(stringFromConfig(config, key, ".output"))
            .seed(intFromConfig(config, key, ".seed", false, (int) System.currentTimeMillis()))
            .numWordsPerFeature(intFromConfig(config, key, ".num_words_per_feature", false))
            .numTokensPerWord(intFromConfig(config, key, ".num_tokens_per_word", false));
  }

  @Override
  protected void setup() {
    // TODO (Brad): I may be introducing a bug here.  Need to confirm.  It's expensive to generate
    // a new Random on every transform. It's also a bit weird because it will produce the same
    // values for every transform this way. Is that intentional? Do we want "deterministic"
    // randomness for some reason?
    rnd = new Random(seed);
    families = familyNames.stream()
        .map(registry::family)
        .collect(Collectors.toList());
    outputFamily = registry.family(outputFamilyName);
  }

  // Generates a permutation of the array and appends it
  // to a given deque.
  private void generatePermutation(int size,
                                   IntArrayFIFOQueue dq) {
    dq.clear();
    int[] permutation = new int[size];
    for (int i = 0; i < size; i++) {
      permutation[i] = i;
    }
    for (int i = 0; i < size; i++) {
      int other = rnd.nextInt(size);
      int tmp = permutation[i];
      permutation[i] = permutation[other];
      permutation[other] = tmp;
    }
    for (int i = 0; i < size; i++) {
      dq.enqueue(permutation[i]);
    }
  }

  private int getToken(IntArrayFIFOQueue dq,
                       double[] features) {
    if (dq.size() < WINDOW_SIZE) {
      generatePermutation(features.length, dq);
    }
    byte largest = 0;
    double largestValue = features[dq.dequeueInt()];
    for (byte i = 1; i < WINDOW_SIZE; i++) {
      double value = features[dq.dequeueInt()];
      if (value > largestValue) {
        largestValue = value;
        largest = i;
      }
    }
    return largest;
  }

  private int getWord(IntArrayFIFOQueue dq,
                      double[] features) {
    int result = 0;
    for (int i = 0; i < numTokensPerWord; i++) {
      result |= getToken(dq, features) << 2 * i;
    }
    return result;
  }

  // Returns the "words" for a feature. Compok is not a word.
  private void getWordsForFeature(FamilyVector vector, Set<String> outputs) {
    if (vector == null) {
      return;
    }
    Preconditions.checkArgument(vector instanceof DenseVector,
                                "Each family in WTAHashTransform must be a DenseVector.");
    double[] features = ((DenseVector)vector).denseArray();
    // We switch from Dequeue<Integer> to IntArrayFIFOQueue to avoid boxing and unboxing ints.
    IntArrayFIFOQueue dq = new IntArrayFIFOQueue(features.length);
    for (int i = 0; i < numWordsPerFeature; i++) {
      String word = vector.family().name() + i + ':' + getWord(dq, features);
      outputs.add(word);
    }
  }

  @Override
  protected void doTransform(MultiFamilyVector featureVector) {
    Set<String> outputs = new HashSet<>();
    for (Family family : families) {
      getWordsForFeature(featureVector.get(family), outputs);
    }

    for (String output : outputs) {
      featureVector.putString(outputFamily.feature(output));
    }
  }
}
