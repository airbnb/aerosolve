/* Author: Hector Yee
 * Thrift schema for ML training data and models
 * To generate use the following command line:
 *  thrift --gen java --out ../java/ MLSchema.thrift
 */

namespace java com.airbnb.aerosolve.core
enum FunctionForm {
  SPLINE,
  LINEAR
}

struct FeatureVector {
  // The first field is the feature family. e.g. "geo"
  // The rest are string feature values. e.g. "SF," CA", "USA"
  // e.g. "geo" -> "San Francisco", "CA", "USA"
  1: optional map<string, set<string>> stringFeatures;

  // The first field is the feature family,
  // the rest is a sparse float feature.
  // e.g "location" -> "lat" : 37.7, "long" : 40.0

  // Labels are a special family of the float features e.g. "$rank"
  // and tranforms can be applied to them to get various models from the same data.
  // For example $rank can be time and one can use the label directly
  // or threshold it for classification for example by a transform.
  2: optional map<string, map<string, double>> floatFeatures;

  // The first field is the feature family, e.g. "image_rgb_histogram"
  // the rest is a dense float feature vector
  3: optional map<string, list<double>> denseFeatures;
}

struct Example {
  // Repeated list of examples in a bag, e.g. groups by user session
  // or ranked list.
  1: optional list<FeatureVector> example;
  // The context feature, e.g. query / user features that is in common
  // over the whole session.
  2: optional FeatureVector context;
}

// The model file would contain a header
// followed by multiple model records.
// The header contains information for the factory
// method to create the model.
struct ModelHeader {
  // e.g. linear, spline
  1: optional string modelType;
  // The number of records following that belong to this model.
  2: optional i64 numRecords;
  // The number of hidden units in neural net models
  3: optional i32 numHidden;
  // calibration parameter
  4: optional double slope;
  5: optional double offset;
  6: optional DictionaryRecord dictionary;
}

struct ModelRecord {
  1: optional ModelHeader modelHeader;
  // e.g. "geo"
  2: optional string featureFamily;
  // e.g. "San Francisco"
  3: optional string featureName;
  // e.g. 1.2
  4: optional double featureWeight;
  // opaque third party serialization
  5: optional string opaque;
  6: optional double scale;
  7: optional list<double> weightVector;
  8: optional double minVal;
  9: optional double maxVal;
  10: optional double threshold;
  11: optional i32 leftChild;
  12: optional i32 rightChild;
  // e.g. SPLINE, LINEAR
  13: optional FunctionForm functionForm;
}

struct EvaluationRecord {
  1: optional double score;
  2: optional double label;
  3: optional bool is_training;
}

struct DebugScoreRecord {
  1: optional string featureFamily;
  2: optional string featureName;
  3: optional double featureValue;
  4: optional double featureWeight;
}

struct DebugScoreDiffRecord {
  1: optional string featureFamily;
  2: optional string featureName;
  3: optional double featureValue1;
  4: optional double featureValue2;
  5: optional double featureWeight1;
  6: optional double featureWeight2;
  7: optional double featureWeightDiff;
}

struct DictionaryEntry {
  1: optional i32 index;
  2: optional double mean;
  3: optional double scale; 
}

struct DictionaryRecord {
  1: optional map<string, map<string, DictionaryEntry>> dictionary;
  2: optional i32 entryCount
}