package org.bdproject;

// Java Libraries
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Map;
import java.util.Iterator;
import java.util.HashMap;
import java.util.*;

// OpenCV Libraries
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.ml.EM;

// LibSVM Libraries
import ca.uwo.csd.ai.nlp.common.SparseVector;
import ca.uwo.csd.ai.nlp.libsvm.svm;
import ca.uwo.csd.ai.nlp.libsvm.svm_model;
import ca.uwo.csd.ai.nlp.libsvm.svm_node;
import ca.uwo.csd.ai.nlp.libsvm.svm_problem;
import ca.uwo.csd.ai.nlp.libsvm.svm_parameter;
import ca.uwo.csd.ai.nlp.libsvm.ex.Instance;

// Spark Libraries
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.*;
import scala.Tuple2;

public class Main {
	
	public static ArrayList< ArrayList<Double>> train_data = new ArrayList< ArrayList<Double>>();
	public static ArrayList< ArrayList<Double>> means = new ArrayList< ArrayList<Double>>();
	public static ArrayList<Integer> labels = new ArrayList<Integer>();
	// global variables
	public static int NUM_SAMPLES = 0;
	public static int DIM = 0;
	public static int NUM_CLASSES = 0;
	public static int K = 2;

	// flag to switch between approaches
	public static boolean flag;

	// flag to enable / disable soft clustering
	public static boolean soft;

	private static final PairFunction<String, Integer, ArrayList<Double>> readData = 
        new PairFunction<String, Integer, ArrayList<Double>>(){
            @Override
            public Tuple2<Integer, ArrayList<Double>> call(String line) throws Exception{
                String cvsSplitBy = ",";
                String[] str = line.split(cvsSplitBy);
				ArrayList<Double> d = new ArrayList<Double>();
				int length = str.length;
				for (int i=0; i<length; i++){
                    d.add(Double.parseDouble(str[i]));
                }
                int cls = d.get(length-1).intValue();
                return new Tuple2(cls, d);
            }
	};

    private static final Function2<ArrayList<Double>, ArrayList<Double>, ArrayList<Double>> compute_sum =
      new Function2<ArrayList<Double>, ArrayList<Double>, ArrayList<Double>>() {
        @Override
        public ArrayList<Double> call(ArrayList<Double> a, ArrayList<Double> b) throws Exception {
          ArrayList<Double> c = new ArrayList<Double>();
          for (int i =0 ; i < a.size(); i++){
              c.add(a.get(i) + b.get(i));
          }
          return c;
        }
      };
      
	private static final Function2<Integer, Integer, Integer> compute_count =
      new Function2<Integer, Integer, Integer>() {
        @Override
        public Integer call(Integer a, Integer b) throws Exception {
        	return a+b;
        }
      };
     private static final PairFunction<Tuple2<Integer, Tuple2<Integer, ArrayList<Double>>>, Integer, ArrayList<Double>> computeMeans =
        new PairFunction<Tuple2<Integer, Tuple2<Integer, ArrayList<Double>>>, Integer, ArrayList<Double>>() {
            @Override
            public Tuple2<Integer, ArrayList<Double>> call(Tuple2<Integer, Tuple2<Integer, ArrayList<Double>>> t) throws Exception {
                ArrayList<Double> avg = new ArrayList<Double>();
                ArrayList<Double> inp = t._2._2;
                int count = t._2._1;
                for (int i=0; i < inp.size(); i++){
                    avg.add(inp.get(i)/count);
                }
                return new Tuple2(t._1, avg);
            }
        };

	public static void printClasses(ArrayList<Integer> cls) {
		for (int i=0; i<cls.size(); i++) { System.out.print(cls.get(i) + " "); }
		System.out.println();
	}

	public static void removeMin(ArrayList<Integer> vec, ArrayList<Double> val) {
		double min = Double.MAX_VALUE;
		int ind = 0;
		for (int i=0; i<val.size(); i++) {
			if (val.get(i) < min) {
				min = val.get(i);
				ind = i;
			}
		}
		vec.remove(ind);
	}

	public static void splitClasses(ArrayList<Integer> cls, ArrayList<Integer> left, ArrayList<Integer> right) {
		if (soft == true) {
			// split classes into 2 groups using EM Algorithm
			Mat label = new Mat();
			Mat points = new Mat(cls.size(), DIM-1, CvType.CV_32F);
			for (int i=0; i<cls.size(); i++) {
				for (int j=0; j<DIM-1; j++) {
					points.put(i, j, means.get(cls.get(i)).get(j));
				}
			}

			Mat probs = new Mat();
			EM em_model = new EM(K, EM.COV_MAT_DIAGONAL, new TermCriteria(TermCriteria.EPS, 10, 0.1));
			em_model.train(points, new Mat(), label, probs);
			ArrayList<Double> l_prob = new ArrayList<Double>();
			ArrayList<Double> r_prob = new ArrayList<Double>();
			
			double ind = 0.0;
			double EM_PROB_THRESHOLD = 0.80;
			for(int i=0; i<probs.rows(); i++) {
				ind = probs.get(i, 0)[0];
				if (ind > EM_PROB_THRESHOLD) { left.add(cls.get(i)); l_prob.add(ind); }
				ind = probs.get(i, 1)[0];
				if (ind > EM_PROB_THRESHOLD) { right.add(cls.get(i)); r_prob.add(ind); }
			}

			if (left.size() == cls.size()) { removeMin(left, l_prob); }
			if (right.size() == cls.size()) { removeMin(right, r_prob); }
		}
		else {
			// split classes into 2 groups using KMeans
			Mat label = new Mat();
			Mat centers = new Mat();
			Mat points = new Mat(cls.size(), DIM-1, CvType.CV_32F);
			for (int i=0; i<cls.size(); i++) {
				for (int j=0; j<DIM-1; j++) {
					points.put(i, j, means.get(cls.get(i)).get(j));
				}
			}
			Core.kmeans(points, K, label, new TermCriteria(TermCriteria.MAX_ITER, 100000, 0.00001), 1000, Core.KMEANS_PP_CENTERS, centers);
			int ind = 0;
			for (int i=0; i<cls.size(); i++) {
				ind = (int)label.get(i, 0)[0];
				if (ind == 0) { left.add(cls.get(i)); }
				else { right.add(cls.get(i)); }
			}
		}
	}

	public static PairFunction<ArrayList<Integer>, ArrayList<Integer>, ArrayList<Integer>> splitClasses = new PairFunction<ArrayList<Integer>, ArrayList<Integer>, ArrayList<Integer>>() {
		public Tuple2<ArrayList<Integer>, ArrayList<Integer>> call(ArrayList<Integer> cls) {
			ArrayList<Integer> left = new ArrayList<Integer>();
			ArrayList<Integer> right = new ArrayList<Integer>();
			if (soft == true) {
				// split classes into 2 groups using EM Algorithm
				Mat label = new Mat();
				Mat points = new Mat(cls.size(), DIM-1, CvType.CV_32F);
				for (int i=0; i<cls.size(); i++) {
					for (int j=0; j<DIM-1; j++) {
						points.put(i, j, means.get(cls.get(i)).get(j));
					}
				}

				Mat probs = new Mat();
				EM em_model = new EM(K, EM.COV_MAT_DIAGONAL, new TermCriteria(TermCriteria.EPS, 10, 0.1));
				em_model.train(points, new Mat(), label, probs);
				ArrayList<Double> l_prob = new ArrayList<Double>();
				ArrayList<Double> r_prob = new ArrayList<Double>();
				
				double ind = 0.0;
				double EM_PROB_THRESHOLD = 0.80;
				for(int i=0; i<probs.rows(); i++) {
					ind = probs.get(i, 0)[0];
					if (ind > EM_PROB_THRESHOLD) { left.add(cls.get(i)); l_prob.add(ind); }
					ind = probs.get(i, 1)[0];
					if (ind > EM_PROB_THRESHOLD) { right.add(cls.get(i)); r_prob.add(ind); }
				}

				if (left.size() == cls.size()) { removeMin(left, l_prob); }
				if (right.size() == cls.size()) { removeMin(right, r_prob); }
			}
			else {
				// split classes into 2 groups using KMeans
				Mat label = new Mat();
				Mat centers = new Mat();
				Mat points = new Mat(cls.size(), DIM-1, CvType.CV_32F);
				for (int i=0; i<cls.size(); i++) {
					for (int j=0; j<DIM-1; j++) {
						points.put(i, j, means.get(cls.get(i)).get(j));
					}
				}
				Core.kmeans(points, K, label, new TermCriteria(TermCriteria.MAX_ITER, 100000, 0.00001), 1000, Core.KMEANS_PP_CENTERS, centers);
				int ind = 0;
				for (int i=0; i<cls.size(); i++) {
					ind = (int)label.get(i, 0)[0];
					if (ind == 0) { left.add(cls.get(i)); }
					else { right.add(cls.get(i)); }
				}
			}
			return new Tuple2(left, right);
		}
	};

	public static void createSVMInput(ArrayList<Integer> left, ArrayList<Integer> right, String input_file) {
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(input_file);
			if (flag == true) { // create input using all samples of the data
				for (int i=0; i<NUM_SAMPLES; i++) {
					int cls = train_data.get(i).get(DIM-1).intValue();
					if (left.contains(cls) && !right.contains(cls)) { pw.print("-1 "); }
					if (!left.contains(cls) && right.contains(cls)) { pw.print("1 "); }
					for (int j=0; j<DIM-1; j++) { pw.print((j+1)+":"+train_data.get(i).get(j)+" "); }
					pw.println();
				}
			}

			else { // create input using only the means of the classes
				// write means of left classes
				for (int i=0; i<left.size(); i++) {
					int cls = left.get(i);
					pw.print("-1 ");
					for (int j=0; j<DIM-1; j++) { pw.print((j+1)+":"+means.get(cls).get(j)+" "); }
					pw.println();
				}

				// write means of right classes
				for (int i=0; i<right.size(); i++) {
					int cls = right.get(i);
					pw.print("1 ");
					for (int j=0; j<DIM-1; j++) { pw.print((j+1)+":"+means.get(cls).get(j)+" "); }
					pw.println();
				}
			}
		}
		catch (FileNotFoundException e) { e.printStackTrace(); }
		finally {
			if (pw != null) {
				try { pw.close(); }
				catch (Exception e) { e.printStackTrace(); }
			}
		}
	}

	public static Instance[] getSVMTrainInput(ArrayList<Integer> left, ArrayList<Integer> right) {
		ArrayList<Double> labels = new ArrayList<Double>();
		ArrayList<SparseVector> vectors = new ArrayList<SparseVector>();

		if (flag == true) { // create input using all samples of the data
			for (int i=0; i<NUM_SAMPLES; i++) {
				int cls = train_data.get(i).get(DIM-1).intValue();
				if (left.contains(cls)) {
					labels.add(-1.0);
					SparseVector vector = new SparseVector(DIM-1);
					for (int j=0; j<DIM-1; j++) { vector.add((j+1), train_data.get(i).get(j)); }
					vectors.add(vector);
				}
				else if (right.contains(cls)) {
					labels.add(1.0);
					SparseVector vector = new SparseVector(DIM-1);
					for (int j=0; j<DIM-1; j++) { vector.add((j+1), train_data.get(i).get(j)); }
					vectors.add(vector);
				}
			}
		}

		else { // create input using only the means of the classes
			// write means of left classes
			for (int i=0; i<left.size(); i++) {
				int cls = left.get(i);
				labels.add(-1.0);
				SparseVector vector = new SparseVector(DIM-1);
				for (int j=0; j<DIM-1; j++) { vector.add((j+1), means.get(cls).get(j)); }
				vectors.add(vector);
			}

			// write means of right classes
			for (int i=0; i<right.size(); i++) {
				int cls = right.get(i);
				labels.add(1.0);
				SparseVector vector = new SparseVector(DIM-1);
				for (int j=0; j<DIM-1; j++) { vector.add((j+1), means.get(cls).get(j)); }
				vectors.add(vector);
			}
		}
		
		Instance[] instances = new Instance[labels.size()];
		for (int i = 0; i < instances.length; i++) {
			instances[i] = new Instance(labels.get(i), vectors.get(i));
		}
		return instances;
	}

	public static svm_problem getSVMProblem(ArrayList<Integer> left, ArrayList<Integer> right) {

		Vector<Double> vy = new Vector<Double>();
		Vector<svm_node[]> vx = new Vector<svm_node[]>();
		int max_index = 0;

		for (int i=0; i<NUM_SAMPLES; i++) {
			int cls = train_data.get(i).get(DIM-1).intValue();
			if (left.contains(cls)) {
				vy.addElement(-1.0);
				svm_node[] x = new svm_node[DIM-1];
				for(int j=0;j<DIM-1;j++) {
					x[j] = new svm_node();
					x[j].index = j+1;
					x[j].value = train_data.get(i).get(j);
				}
				if(DIM-1>0) max_index = Math.max(max_index, x[DIM-2].index);
				vx.addElement(x);
			}
			else if (right.contains(cls)) {
				vy.addElement(1.0);
				svm_node[] x = new svm_node[DIM-1];
				for(int j=0;j<DIM-1;j++) {
					x[j] = new svm_node();
					x[j].index = j+1;
					x[j].value = train_data.get(i).get(j);
				}
				if(DIM-1>0) max_index = Math.max(max_index, x[DIM-2].index);
				vx.addElement(x);
			}
		}
		svm_problem prob = new svm_problem();
		prob.l = vy.size();
		prob.x = new svm_node[prob.l][];
		for(int i=0;i<prob.l;i++)
		prob.x[i] = vx.elementAt(i);
		prob.y = new double[prob.l];
		for(int i=0;i<prob.l;i++)
		prob.y[i] = vy.elementAt(i);
		return prob;
	}

	public static Node constructBHDT(Node tree, ArrayList<Integer> labels) {
		if (labels.size() > 1) {
			System.out.println("------------------------------------------------------------");
			System.out.println("CLASS LABELS IN THE NODE:");
			printClasses(labels);

			// split classes into two groups
			ArrayList<Integer> left = new ArrayList<Integer>();
			ArrayList<Integer> right = new ArrayList<Integer>();
			splitClasses(labels, left, right);

			System.out.println("LEFT: ");
			printClasses(left);

			System.out.println("RIGHT: ");
			printClasses(right);

			// train a binary svm
			System.out.println("Training a Binary SVM");
			Instance[] train_samples = getSVMTrainInput(left, right);
			svm_problem prob = getSVMProblem(left, right);
			svm_parameter param = new svm_parameter();
			param.svm_type = svm_parameter.C_SVC;
			param.kernel_type = svm_parameter.LINEAR;
			param.C = 10; // by trial and error
			svm_model model = svm.svm_train(prob, param);

			tree = new Node();
			tree.classes = new ArrayList<Integer>();
			for(int i=0; i<labels.size(); i++) tree.classes.add(labels.get(i));
			tree.model = model;
			tree.left = constructBHDT(tree.left, left);
			tree.right = constructBHDT(tree.right, right);
		}
		else if (labels.size() == 1) {
			tree = new Node();
			tree.classes = new ArrayList<Integer>();
			tree.classes.add(labels.get(0));
			tree.left = null;
			tree.right = null;
		}
		return tree;
	}

	public static Instance getSVMTestInput(double[] d) {
		
		SparseVector vector = new SparseVector(DIM-1);
		for (int j=0; j<DIM-1; j++) { vector.add((j+1), d[j]); }
		return new Instance(-1, vector);
	}
	public static int predict(svm_model model, double[] d) {
		
		svm_node[] svm_nodes = new svm_node[DIM-1];
		for (int j=0; j<svm_nodes.length; j++) { svm_nodes[j] = new svm_node(j+1, d[j]); }
		double p = svm.svm_predict(model, svm_nodes);
		return (int)p;
	}

	public static int classifySample(Node tree, double[] d) {
		if (tree != null) {
			if (tree.classes.size() == 1) return tree.classes.get(0);
			if (predict(tree.model, d) == -1) return classifySample(tree.left, d);
			else return classifySample(tree.right, d);
		}
		return -1;
	}

	public static double[] convertToArray(ArrayList<Double> d) {
		double[] t = new double[d.size()];
		 for (int i = 0; i < t.length; i++) { t[i] = d.get(i).doubleValue(); }
		 return t;
	}
	
	public static void test(Node tree, JavaRDD<String> scTestFile) {
		JavaPairRDD<Integer, ArrayList<Double>> _test_data = scTestFile.mapToPair(readData);
		ArrayList<ArrayList<Double>> test_data = new ArrayList(_test_data.values().collect());

		int correct = 0;
		int total = test_data.size();
		int length = test_data.get(0).size();

		for(int i=0; i<total; i++) {
			ArrayList<Double> d = test_data.get(i);
			int label = d.get(length-1).intValue();
			int cls = classifySample(tree, convertToArray(d));
			if (label == cls) { correct++; }
		}
		System.out.println("************************************************************************************");
		System.out.println("CORRECTLY CLASSIFIED SAMPLES = " + correct);
		System.out.println("TOTAL SAMPLES = " + total);
		double accuracy = (double)correct / total;
		System.out.println("ACCURACY = " + (accuracy*100) + " %");
	}

	public static Function2<Node, String, Boolean> test = new Function2<Node, String, Boolean>() {

		public Boolean call(Node tree, String test_file) {
			BufferedReader br = null;
			String line = "";
			String cvsSplitBy = ",";
			try {
				int correct = 0;
				int total = 0;

				br = new BufferedReader(new FileReader(test_file));
				while ((line = br.readLine()) != null) {
					String[] str = line.split(cvsSplitBy);
					ArrayList<Double> d = new ArrayList<Double>();
					int length = str.length;
					for(int i=0; i<length; i++) { d.add(Double.parseDouble(str[i])); }
					int label = d.get(length-1).intValue();
					int cls = classifySample(tree, convertToArray(d));
					if (label == cls) { correct++; }
					total++;
				}
				System.out.println("************************************************************************************");
				System.out.println("CORRECTLY CLASSIFIED SAMPLES = " + correct);
				System.out.println("TOTAL SAMPLES = " + total);
				double accuracy = (double)correct / total;
				System.out.println("ACCURACY = " + (accuracy*100) + " %");
			}
			catch (FileNotFoundException e) { e.printStackTrace(); }
			catch (IOException e) { e.printStackTrace(); }
			finally {
				if (br != null) {
					try { br.close(); }
					catch (IOException e) { e.printStackTrace(); }
				}
			}
			return true;
		}
	};

	public static void main(String[] args) {
		
		// load OpenCV library
		System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
	    
		if (args.length != 2) {
			System.out.println("-----------------------------------------------------------------------------------");
			System.out.println("Usage:");
			System.out.println("./bhdt <train_path> <test_path>");
			System.out.println("-----------------------------------------------------------------------------------");
			System.exit(0);
		}
        
        SparkConf conf = new SparkConf().setAppName("org.bdproject.bhdt").setMaster("local");
        conf.set("spark.driver.maxResultSize", "5g");
        JavaSparkContext jsc = new JavaSparkContext(conf);

		String train_file = args[0];
		String test_file = args[1];
        
        JavaRDD<String> scTrainFile = jsc.textFile(train_file);
        JavaRDD<String> scTestFile = jsc.textFile(test_file);

		// flag to switch between approaches
		flag = true;

		// flag to enable / disable soft clustering
		soft = false;

		// read the training data
		System.out.println("************************************************************************************");
		System.out.println("READING TRAINING DATA");
        JavaPairRDD<Integer, ArrayList<Double>> _train_data = scTrainFile.mapToPair(readData);
        JavaPairRDD<Integer, ArrayList<Double>> _train_data_sum  = _train_data.reduceByKey(compute_sum);
        JavaPairRDD<Integer, Integer> class_count = _train_data.mapToPair(f -> {return new Tuple2(f._1,1);}).reduceByKey(compute_count);
        JavaPairRDD<Integer, ArrayList<Double>> _means = class_count.join(_train_data_sum).mapToPair(computeMeans);
        train_data = new ArrayList(_train_data.values().collect());
        means = new ArrayList(_means.values().collect());
		System.out.println("************************************************************************************");

		NUM_SAMPLES = (int)_train_data.count();
		DIM = _train_data.first()._2.size();
		NUM_CLASSES = (int)_train_data.keys().distinct().count();

		System.out.println("NUM_SAMPLES = " + NUM_SAMPLES);
		System.out.println("DIM = " + DIM);
		System.out.println("NUM_CLASSES = " + NUM_CLASSES);

		ArrayList labels = new ArrayList(_train_data.keys().distinct().collect());
		// train a BHDT
		System.out.println("************************************************************************************");
		System.out.println("TRAINING A BINARY HIERARCHICAL DECISION TREE");
		Node tree = null;
		tree = constructBHDT(tree, labels);

		// test using BHDT
		System.out.println("************************************************************************************");
		System.out.println("TESTING USING BHDT");
		test(tree, scTestFile);
	}
}