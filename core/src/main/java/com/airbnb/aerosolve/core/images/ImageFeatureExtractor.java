package com.airbnb.aerosolve.core.images;

import com.airbnb.aerosolve.core.perf.FastMultiFamilyVector;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import java.awt.image.BufferedImage;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/*
  Calls all known features and adds them as dense features in a feature vector.
 */
public class ImageFeatureExtractor implements Serializable {
  final List<ImageFeature> features;

  private static final ThreadLocal<ImageFeatureExtractor> EXTRACTOR =
      new ThreadLocal<ImageFeatureExtractor>() {
    @Override protected ImageFeatureExtractor initialValue() {
      return new ImageFeatureExtractor();
    }
  };

  public static ImageFeatureExtractor getInstance() {
    return EXTRACTOR.get();
  }

  private ImageFeatureExtractor() {
    features = new ArrayList<ImageFeature>();
    features.add(new RGBFeature());
    features.add(new HOGFeature());
    features.add(new LBPFeature());
    features.add(new HSVFeature());
  }

  public MultiFamilyVector getFeatureVector(BufferedImage image, FeatureRegistry registry) {
    MultiFamilyVector featureVector = new FastMultiFamilyVector(registry);
    for (ImageFeature feature : features) {
      List<Float> values = feature.extractFeatureSPMK(image);
      double[] dblValues = new double[values.size()];
      for (int i = 0; i < values.size(); i++) {
        dblValues[i] = values.get(i);
      }
      featureVector.putDense(registry.family(feature.featureName()), dblValues);
    }
    return featureVector;
  }
}
