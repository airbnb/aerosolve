namespace java com.airbnb.aerosolve.core

// Thrift schema for KDTrees

enum KDTreeNodeType {
  X_SPLIT = 1,
  Y_SPLIT = 2,
  LEAF = 3
}

struct KDTreeNode {
  1: optional KDTreeNodeType nodeType;
  2: optional double splitValue;
  3: optional i32 leftChild;
  4: optional i32 rightChild;
  5: optional double minX;
  6: optional double maxX;
  7: optional double minY;
  8: optional double maxY;
  9: optional i32 count;
}