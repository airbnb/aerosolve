package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.KDTreeNode;
import com.airbnb.aerosolve.core.KDTreeNodeType;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class KDTreeModelTest {
  private static final Logger log = LoggerFactory.getLogger(KDTreeModelTest.class);

  //                    4
  //         |--------------- y = 2
  //  1      | 2       3
  //     x = 1
  public static KDTreeNode[] getTestNodes() {
    KDTreeNode parent = new KDTreeNode();
    parent.setNodeType(KDTreeNodeType.X_SPLIT);
    parent.setSplitValue(1.0);
    parent.setLeftChild(1);
    parent.setRightChild(2);

    KDTreeNode one = new KDTreeNode();
    one.setNodeType(KDTreeNodeType.LEAF);

    KDTreeNode two = new KDTreeNode();
    two.setNodeType(KDTreeNodeType.Y_SPLIT);
    two.setSplitValue(2.0);
    two.setLeftChild(3);
    two.setRightChild(4);

    KDTreeNode three = new KDTreeNode();
    three.setNodeType(KDTreeNodeType.LEAF);

    KDTreeNode four = new KDTreeNode();
    four.setNodeType(KDTreeNodeType.LEAF);

    KDTreeNode[] arr = {parent, one, two, three, four};
    return arr;

  }

  @Test
  public void testLeaf() {
    KDTreeModel tree = new KDTreeModel(getTestNodes());

    int leaf = tree.leaf(-1.0, -1.0);
    assertEquals(1, leaf);

    leaf = tree.leaf(1.1, 0.0);
    assertEquals(3, leaf);

    leaf = tree.leaf(1.0, 2.0);
    assertEquals(4, leaf);

    leaf = tree.leaf(0.99, 2.1);
    assertEquals(1, leaf);
  }

  @Test
  public void testQuery() {
    KDTreeModel tree = new KDTreeModel(getTestNodes());

    ArrayList<Integer> res1 = tree.query(-1.0, -1.0);
    assertEquals(2, res1.size());
    assertEquals(0, res1.get(0).intValue());
    assertEquals(1, res1.get(1).intValue());

    ArrayList<Integer> res2 = tree.query(1.1, 0.0);
    assertEquals(3, res2.size());
    assertEquals(0, res2.get(0).intValue());
    assertEquals(2, res2.get(1).intValue());
    assertEquals(3, res2.get(2).intValue());

    ArrayList<Integer> res3 = tree.query(1.0, 2.0);
    assertEquals(3, res3.size());
    assertEquals(0, res3.get(0).intValue());
    assertEquals(2, res3.get(1).intValue());
    assertEquals(4, res3.get(2).intValue());

    ArrayList<Integer> res4 = tree.query(0.99, 2.1);
    assertEquals(2, res4.size());
    assertEquals(0, res4.get(0).intValue());
    assertEquals(1, res4.get(1).intValue());
  }

  private Set<Integer> toSet(ArrayList<Integer> arr) {
    Set<Integer> set = new HashSet<>();
    for (Integer i : arr) {
      set.add(i);
    }
    return set;
  }

  @Test
  public void testQueryBox() {
    KDTreeModel tree = new KDTreeModel(getTestNodes());

    // This box covers all nodes
    Set<Integer> q1 = toSet(tree.queryBox(-10.0, -10.0, 10.0, 10.0));
    assertEquals(q1.size(), 5);
    assertTrue(q1.contains(0));
    assertTrue(q1.contains(1));
    assertTrue(q1.contains(2));
    assertTrue(q1.contains(3));
    assertTrue(q1.contains(4));

    // This box covers only nodes 0 and 1
    Set<Integer> q2 = toSet(tree.queryBox(-10.0, -10.0, 0.9, 10.0));
    assertEquals(q2.size(), 2);
    assertTrue(q2.contains(0));
    assertTrue(q2.contains(1));

    // This box covers only nodes 0, 1, 2, 4
    Set<Integer> q3 = toSet(tree.queryBox(-10.0, 2.0, 2.0, 10.0));
    assertEquals(q3.size(), 4);
    assertTrue(q3.contains(0));
    assertTrue(q3.contains(1));
    assertTrue(q3.contains(2));
    assertTrue(q3.contains(4));

    // This box covers only node 0, 2, 3
    Set<Integer> q4 = toSet(tree.queryBox(1.1, 0.0, 10.0, 1.9));
    assertEquals(q4.size(), 3);
    assertTrue(q4.contains(0));
    assertTrue(q4.contains(2));
    assertTrue(q4.contains(3));

    // This box covers all nodes except 1
    Set<Integer> q5 = toSet(tree.queryBox(1.0, 0.0, 10.0, 2.0));
    assertEquals(q5.size(), 4);
    assertTrue(q5.contains(0));
    assertTrue(q5.contains(2));
    assertTrue(q5.contains(3));
    assertTrue(q5.contains(4));
  }
}