package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.NDTreeNode;
import com.airbnb.aerosolve.core.util.Util;
import com.google.common.base.Optional;
import lombok.Getter;
import org.apache.commons.codec.binary.Base64;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

/*
  N-Dimensional KDTreeModel.
 */
public class NDTreeModel implements Serializable {
  private static final long serialVersionUID = -2884260218927875615L;
  private static final Logger log = LoggerFactory.getLogger(NDTreeModel.class);
  public static final int LEAF = -1;
  @Getter
  private NDTreeNode[] nodes;

  public NDTreeModel(NDTreeNode[] node) {
    nodes = node;
  }

  public NDTreeModel(List<NDTreeNode> nodeList) {
    nodes = new NDTreeNode[nodeList.size()];
    nodeList.toArray(nodes);
  }

  public int leaf(float ... coordinates) {
    if (nodes == null || nodes.length == 0) return -1;
    int currIdx = 0;

    while (true) {
      int nextIdx = next(currIdx, coordinates);
      if (nextIdx == -1) {
        return currIdx;
      } else {
        currIdx = nextIdx;
      }
    }
  }
    // Returns the indice of leaf containing the point.
  public int leaf(List<Float> coordinates) {
    if (nodes == null || nodes.length == 0) return -1;
    int currIdx = 0;

    while (true) {
      int nextIdx = next(currIdx, coordinates);
      if (nextIdx == -1) {
        return currIdx;
      } else {
        currIdx = nextIdx;
      }
    }
  }

  private int next(int currIdx, float[] coordinates) {
    NDTreeNode node = nodes[currIdx];
    int index = node.coordinateIndex;
    if (index == LEAF) {
      // leaf
      return -1;
    } else {
      if (coordinates[index] < node.splitValue) {
        return node.leftChild;
      } else {
        return node.rightChild;
      }
    }
  }

  private int next(int currIdx, List<Float> coordinates) {
    NDTreeNode node = nodes[currIdx];
    int index = node.coordinateIndex;
    if (index == LEAF) {
      // leaf
      return -1;
    } else {
      if (coordinates.get(index) < node.splitValue) {
        return node.leftChild;
      } else {
        return node.rightChild;
      }
    }
  }

  public NDTreeNode getNode(int id) {
    return nodes[id];
  }

  // Returns the indices of nodes traversed to get to the leaf containing the point.
  public ArrayList<Integer> query(List<Float> coordinates) {
    ArrayList<Integer> idx = new ArrayList<>();

    if (nodes == null) return idx;

    int currIdx = 0;
    while (true) {
      idx.add(currIdx);
      int nextIdx = next(currIdx, coordinates);
      if (nextIdx == -1) {
        return idx;
      } else {
        currIdx = nextIdx;
      }
    }
  }

  public ArrayList<Integer> query(float ... coordinates) {
    ArrayList<Integer> idx = new ArrayList<>();

    if (nodes == null) return idx;

    int currIdx = 0;
    while (true) {
      idx.add(currIdx);
      int nextIdx = next(currIdx, coordinates);
      if (nextIdx == -1) {
        return idx;
      } else {
        currIdx = nextIdx;
      }
    }
  }

  // Returns the indices of all node overlapping the box
  public ArrayList<Integer> queryBox(List<Double> min, List<Double> max) {
    ArrayList<Integer> idx = new ArrayList<>();
    assert (min.size() == max.size());
    if (nodes == null) return idx;

    Stack<Integer> stack = new Stack<Integer>();
    stack.push(0);
    while (!stack.isEmpty()) {
      int currIdx = stack.pop();
      idx.add(currIdx);
      NDTreeNode node = nodes[currIdx];
      int index = node.coordinateIndex;
      if (index > LEAF) {
        if (min.get(index) < node.splitValue) {
          stack.push(node.leftChild);
        }
        if (max.get(index) >= node.splitValue) {
          stack.push(node.rightChild);
        }
      }
    }
    return idx;
  }
  
  public static Optional<NDTreeModel> readFromGzippedStream(InputStream inputStream) {
    List<NDTreeNode> nodes = Util.readFromGzippedStream(NDTreeNode.class, inputStream);
    if (!nodes.isEmpty()) {
      return Optional.of(new NDTreeModel(nodes));
    } else {
      return Optional.absent();
    }
  }

  public static Optional<NDTreeModel> readFromGzippedResource(String name) {
    InputStream inputStream = java.lang.ClassLoader.getSystemResourceAsStream(name);
    Optional<NDTreeModel> modelOptional = readFromGzippedStream(inputStream);
    if (!modelOptional.isPresent()) {
      log.error("Could not load resource named " + name);
    }
    return modelOptional;
  }

  public static Optional<NDTreeModel> readFromGzippedBase64String(String encoded) {
    byte[] decoded = Base64.decodeBase64(encoded);
    InputStream stream = new ByteArrayInputStream(decoded);
    return readFromGzippedStream(stream);
  }

}
