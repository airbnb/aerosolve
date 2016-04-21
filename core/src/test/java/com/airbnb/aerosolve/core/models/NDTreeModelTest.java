package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.NDTreeNode;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static com.airbnb.aerosolve.core.models.NDTreeModel.LEAF;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class NDTreeModelTest {
  private static final Logger log = LoggerFactory.getLogger(NDTreeModelTest.class);
  //                    4
  //         |--------------- y = 2
  //  1      | 2       3
  //     x = 1
  // total space is from (0,0) to (4,4)
  public static NDTreeModel getNDTreeModel() {
    NDTreeNode parent = new NDTreeNode();
    parent.setCoordinateIndex(0);
    parent.setSplitValue(1.0);
    parent.setLeftChild(1);
    parent.setRightChild(2);

    NDTreeNode one = new NDTreeNode();
    one.setCoordinateIndex(LEAF);
    one.setMin(Arrays.asList(0.0,0.0));
    one.setMax(Arrays.asList(1.0,4.0));

    NDTreeNode two = new NDTreeNode();
    two.setCoordinateIndex(1);
    two.setSplitValue(2.0);
    two.setLeftChild(3);
    two.setRightChild(4);

    NDTreeNode three = new NDTreeNode();
    three.setCoordinateIndex(LEAF);
    three.setMin(Arrays.asList(1.0,0.0));
    three.setMax(Arrays.asList(4.0,2.0));

    NDTreeNode four = new NDTreeNode();
    four.setCoordinateIndex(LEAF);
    four.setMin(Arrays.asList(1.0,2.0));
    four.setMax(Arrays.asList(4.0,4.0));

    NDTreeNode[] arr = {parent, one, two, three, four};
    return new NDTreeModel(arr);

  }
  @Test
  public void testLeaf() {
    NDTreeModel tree = getNDTreeModel();

    int leaf = tree.leaf(-1, -1);
    assertEquals(1, leaf);

    leaf = tree.leaf(Arrays.asList((float) -1, (float) -1));
    assertEquals(1, leaf);

    leaf = tree.leaf((float)1.1, (float)0.0);
    assertEquals(3, leaf);
    leaf = tree.leaf(Arrays.asList((float)1.1, (float)0.0));
    assertEquals(3, leaf);

    leaf = tree.leaf((float)1.0, (float)2.0);
    assertEquals(4, leaf);
    leaf = tree.leaf(Arrays.asList((float)1.0, (float)2.0));
    assertEquals(4, leaf);

    leaf = tree.leaf((float)0.99, (float)2.1);
    assertEquals(1, leaf);
    leaf = tree.leaf(Arrays.asList((float)0.99, (float)2.1));
    assertEquals(1, leaf);
  }

  @Test
  public void testQuery() {
    NDTreeModel tree = getNDTreeModel();

    ArrayList<Integer> res1 = tree.query((float)-1.0, (float)-1.0);
    assertEquals(2, res1.size());
    assertEquals(0, res1.get(0).intValue());
    assertEquals(1, res1.get(1).intValue());
    tree.query(Arrays.asList((float)-1.0, (float)-1.0));
    assertEquals(2, res1.size());
    assertEquals(0, res1.get(0).intValue());
    assertEquals(1, res1.get(1).intValue());

    ArrayList<Integer> res2 = tree.query((float)1.1, (float)0.0);
    assertEquals(3, res2.size());
    assertEquals(0, res2.get(0).intValue());
    assertEquals(2, res2.get(1).intValue());
    assertEquals(3, res2.get(2).intValue());
    res2 = tree.query(Arrays.asList((float)1.1, (float)0.0));
    assertEquals(3, res2.size());
    assertEquals(0, res2.get(0).intValue());
    assertEquals(2, res2.get(1).intValue());
    assertEquals(3, res2.get(2).intValue());

    ArrayList<Integer> res3 = tree.query((float)1.0, (float)2.0);
    assertEquals(3, res3.size());
    assertEquals(0, res3.get(0).intValue());
    assertEquals(2, res3.get(1).intValue());
    assertEquals(4, res3.get(2).intValue());
    res3 = tree.query(Arrays.asList((float)1.0, (float)2.0));
    assertEquals(3, res3.size());
    assertEquals(0, res3.get(0).intValue());
    assertEquals(2, res3.get(1).intValue());
    assertEquals(4, res3.get(2).intValue());

    ArrayList<Integer> res4 = tree.query((float)0.99, (float)2.1);
    assertEquals(2, res4.size());
    assertEquals(0, res4.get(0).intValue());
    assertEquals(1, res4.get(1).intValue());
    res4 = tree.query(Arrays.asList((float)0.99, (float)2.1));
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
    NDTreeModel tree = getNDTreeModel();

    // This box covers all nodes
    Set<Integer> q1 = toSet(tree.queryBox(Arrays.asList(-10.0, -10.0), Arrays.asList(10.0, 10.0)));
    assertEquals(q1.size(), 5);
    assertTrue(q1.contains(0));
    assertTrue(q1.contains(1));
    assertTrue(q1.contains(2));
    assertTrue(q1.contains(3));
    assertTrue(q1.contains(4));

    // This box covers only nodes 0 and 1
    Set<Integer> q2 = toSet(tree.queryBox(Arrays.asList(-10.0, -10.0), Arrays.asList(0.9, 10.0)));
    assertEquals(q2.size(), 2);
    assertTrue(q2.contains(0));
    assertTrue(q2.contains(1));

    // This box covers only nodes 0, 1, 2, 4
    Set<Integer> q3 = toSet(tree.queryBox(Arrays.asList(-10.0, 2.0), Arrays.asList(2.0, 10.0)));
    assertEquals(q3.size(), 4);
    assertTrue(q3.contains(0));
    assertTrue(q3.contains(1));
    assertTrue(q3.contains(2));
    assertTrue(q3.contains(4));

    // This box covers only node 0, 2, 3
    Set<Integer> q4 = toSet(tree.queryBox(Arrays.asList(1.1, 0.0), Arrays.asList(10.0, 1.9)));
    assertEquals(q4.size(), 3);
    assertTrue(q4.contains(0));
    assertTrue(q4.contains(2));
    assertTrue(q4.contains(3));

    // This box covers all nodes except 1
    Set<Integer> q5 = toSet(tree.queryBox(Arrays.asList(1.0, 0.0), Arrays.asList(10.0, 2.0)));
    assertEquals(q5.size(), 4);
    assertTrue(q5.contains(0));
    assertTrue(q5.contains(2));
    assertTrue(q5.contains(3));
    assertTrue(q5.contains(4));
  }

}