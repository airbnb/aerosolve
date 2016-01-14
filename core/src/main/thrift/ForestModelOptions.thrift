namespace java com.airbnb.aerosolve.core

struct ForestModelOptions {
  // Number of samples needed to build a tree
  1: optional i32 sampleSize = 100;
  // Maximum depth of a tree
  2: optional i32 maxDepth = 10;
  // Minimum count of items in a leaf.
  3: optional i32 minItemCount = 5;
  // Number of tries to find a split
  4: optional i32 numTries = 10;
}