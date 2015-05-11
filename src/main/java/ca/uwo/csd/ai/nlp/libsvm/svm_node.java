package ca.uwo.csd.ai.nlp.libsvm;
public class svm_node implements java.io.Serializable
{
	public int index;
	public double value;
	public svm_node(int i, double v) {
		index = i;
		value = v;
	}
	public svm_node() {
		index = 1;
		value = 1;
	}
}
