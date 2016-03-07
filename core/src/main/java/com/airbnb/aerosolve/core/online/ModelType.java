package com.airbnb.aerosolve.core.online;

import lombok.Getter;

public class ModelType {
  @Getter
  private final String configName;
  @Getter
  private final String key;

  @Getter
  private final String path;

  public ModelType(String configName, String key, String path) {
    this.configName = configName;
    this.key = key;
    this.path = path;
  }
}
