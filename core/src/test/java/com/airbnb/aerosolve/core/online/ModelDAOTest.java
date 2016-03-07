package com.airbnb.aerosolve.core.online;

import com.sun.javafx.scene.shape.PathUtils;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;

public class ModelDAOTest {
  static public BufferedReader readResource(String name) {
    InputStream stream = PathUtils.class.getResourceAsStream(name);
    return new BufferedReader(new InputStreamReader(stream));
  }

  @Test
  public void rawProbability() throws Exception {
    ModelType incomeModel = new ModelType(
        "income_prediction.conf", "spline_model", "/income.model");
    BufferedReader reader = readResource(incomeModel.getPath());
    FeatureMapping featureMapping = new FeatureMapping();
    featureMapping.add();
    featureMapping.finish();
    ModelDAO modelDAO = new ModelDAO()
  }
}