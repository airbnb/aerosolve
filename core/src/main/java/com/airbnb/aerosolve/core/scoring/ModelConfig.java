package com.airbnb.aerosolve.core.scoring;

import lombok.Getter;
import lombok.experimental.Builder;

@Builder
public class ModelConfig {
  @Getter
  private final String configName;

  @Getter
  private final String key;

  @Getter
  private final String modelName;
}