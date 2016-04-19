namespace java com.airbnb.aerosolve.core

struct NDTreeNode {
// coordinateIndex = min/max size is child,
// coordinateIndex from 0 to min.size()-1 means split in that coordinate
// similar to KDTreeNode's X_SPLICT, Y_SPLICT
  1: optional i32 coordinateIndex;
  2: optional i32 leftChild;
  3: optional i32 rightChild;
  4: optional i32 count;
  5: optional list<double> min;
  6: optional list<double> max;
  7: optional double splitValue;
}