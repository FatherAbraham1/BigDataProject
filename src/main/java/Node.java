package org.bdproject;

import java.util.ArrayList;

import ca.uwo.csd.ai.nlp.libsvm.svm_model;

public class Node {
	public ArrayList<Integer> classes;
	public svm_model model;

	public Node left;
	public Node right;
}