import java.io.*;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

class DetectFaceDemo {
	public void run() {
		System.out.println("\nRunning DetectFaceDemo");

		// read input image
		Mat image = Highgui.imread(getClass().getResource("/face.png").getPath());

		// convert image to grayscale
		Mat grayImage = new Mat(image.rows(), image.cols(), image.type());
		Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGRA2GRAY);
		Core.normalize(grayImage, grayImage, 0, 255, Core.NORM_MINMAX);

		FeatureDetector surfDetector = FeatureDetector.create(FeatureDetector.SURF);
		DescriptorExtractor surfExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
		MatOfKeyPoint keyPoint = new MatOfKeyPoint();
		Mat descriptors = new Mat(image.rows(), image.cols(), image.type());

		// detect key points in the image using SURF
		surfDetector.detect(grayImage, keyPoint);

		// compute SURF features for the keypoints
		surfExtractor.compute(grayImage, keyPoint, descriptors);

		// write features to file
		writeToFile(descriptors, new File("output_test.txt"));
	}

	public void writeToFile(Mat m, File file) {

		try {
			PrintWriter pw = new PrintWriter(file);
			System.out.println("[INFO] Feature Dimension => " + m.rows() + "*" + m.cols() + " = " + (m.rows()*m.cols()));
			pw.println(m.dump());
			pw.close();
		} catch(IOException e) {
			System.out.println("[ERROR] Cannot write to file");
		}
	}
}

public class HelloOpenCV {
	public static void main(String[] args) {
		System.out.println("Hello, OpenCV");

		// Load the native library.
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		new DetectFaceDemo().run();
	}
}