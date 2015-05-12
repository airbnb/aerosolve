package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.KDTreeNode;
import com.airbnb.aerosolve.core.KDTreeNodeType;
import com.airbnb.aerosolve.core.util.Util;
import com.google.common.base.Optional;
import lombok.Getter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.commons.codec.binary.Base64;

import java.io.*;
import java.util.ArrayList;
import java.util.Stack;
import java.util.zip.GZIPInputStream;

// A specialized 2D kd-tree that supports point and box queries.
public class KDTreeModel implements Serializable {

  private static final long serialVersionUID = -2884260218927875695L;
  private static final Logger log = LoggerFactory.getLogger(KDTreeModel.class);

  @Getter
  private KDTreeNode[] nodes;

  public KDTreeModel(KDTreeNode[] node) {
    nodes = node;
  }

  // Returns the indices of nodes traversed to get to the leaf containing the point.
  public ArrayList<Integer> query(double x, double y) {
    ArrayList<Integer> idx = new ArrayList<>();

    if (nodes == null) return idx;

    int currIdx = 0;

    while (currIdx >= 0) {
      KDTreeNode node = nodes[currIdx];
      idx.add(currIdx);
      switch(node.nodeType) {
        case X_SPLIT: {
          if (x < node.splitValue) {
            currIdx = node.leftChild;
          } else {
            currIdx = node.rightChild;
          }
        }
        break;
        case Y_SPLIT: {
          if (y < node.splitValue) {
            currIdx = node.leftChild;
          } else {
            currIdx = node.rightChild;
          }
        }
        break;
        case LEAF: {
          currIdx = -1;
        }
        break;
      }
    }

    return idx;
  }

  // Returns the indices of all node overlapping the box
  public ArrayList<Integer> queryBox(double minX, double minY, double maxX, double maxY) {
    ArrayList<Integer> idx = new ArrayList<>();

    if (nodes == null) return idx;

    Stack<Integer> stack = new Stack<Integer>();
    stack.push(0);
    while (!stack.isEmpty()) {
      int currIdx = stack.pop();
      idx.add(currIdx);
      KDTreeNode node = nodes[currIdx];
      switch (node.nodeType) {
        case X_SPLIT: {
          if (minX < node.splitValue) {
            stack.push(node.leftChild);
          }
          if (maxX >= node.splitValue) {
            stack.push(node.rightChild);
          }
        }
        break;
        case Y_SPLIT: {
          if (minY < node.splitValue) {
            stack.push(node.leftChild);
          }
          if (maxY >= node.splitValue) {
            stack.push(node.rightChild);
          }
        }
        case LEAF:
          break;
      }
    }
    return idx;
  }

  public static Optional<KDTreeModel> readFromGzippedStream(InputStream inputStream) {
    try {
      if (inputStream != null) {
        GZIPInputStream gzipInputStream = new GZIPInputStream(inputStream);
        BufferedReader reader = new BufferedReader(new InputStreamReader(gzipInputStream));
        ArrayList<KDTreeNode> nodes = new ArrayList<>();
        String line = reader.readLine();
        while(line != null) {
          KDTreeNode node = Util.decodeKDTreeNode(line);
          nodes.add(node);
          line = reader.readLine();
        }
        if (!nodes.isEmpty()) {
          KDTreeNode[] array = new KDTreeNode[nodes.size()];
          array = nodes.toArray(array);
          KDTreeModel model  = new KDTreeModel(array);
          return Optional.of(model);
        }
      }
    } catch (IOException e) {
      log.error(e.getMessage());
    }
    return Optional.absent();
  }

  public static Optional<KDTreeModel> readFromGzippedResource(String name) {
    InputStream inputStream = java.lang.ClassLoader.getSystemResourceAsStream(name);
    Optional<KDTreeModel> modelOptional = readFromGzippedStream(inputStream);
    if (!modelOptional.isPresent()) {
      log.error("Could not load resource named " + name);
    }
    return modelOptional;
  }

  public static Optional<KDTreeModel> readFromGzippedBase64String(String encoded) {
    byte[] decoded = Base64.decodeBase64(encoded);
    InputStream stream = new ByteArrayInputStream(decoded);
    return readFromGzippedStream(stream);
  }

  // Strips nodes for queries. To save space we just store the minimum amount of data.
  public static KDTreeNode stripNode(KDTreeNode node) {
    KDTreeNode newNode = new KDTreeNode();
    if (node.isSetNodeType()) {
      newNode.setNodeType(node.nodeType);
    }
    if (node.isSetSplitValue()) {
      newNode.setSplitValue(node.splitValue);
    }
    if (node.isSetLeftChild()) {
      newNode.setLeftChild(node.leftChild);
    }
    if (node.isSetRightChild()) {
      newNode.setRightChild(node.rightChild);
    }
    return newNode;
  }
}
