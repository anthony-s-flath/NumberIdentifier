import mnist.MnistReader;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;


public class TrainingData {

    public int dataAmount;
    public static final int pixelAmount = 28*28;
    public static final int outputAmount = 10;

    private static String TRAINING_LABEL_FILE = "";
    private static String TRAINING_IMAGE_FILE = "";
    private static String TESTING_LABEL_FILE = "";
    private static String TESTING_IMAGE_FILE = "";

    // used for formatting
    // returns list of training data
    // each element is a training datum
    // each datum is an array containing two arrays
    // element 0 is activation of input, with a size of 28*28, value of 0 to 1
    // element 1 has expected output, size 10 ( 0 - 9 ), value of either 0 or 1
    public List<SimpleMatrix[]> getData(boolean isTraining) {
        MnistReader mnistReader = new MnistReader();
        int[] labels;
        // has 2d array of pixels 0 - 255
        List<int[][]> imageList;

        if (isTraining) {
            // has 0 - 9
            labels = mnistReader.getLabels(TRAINING_LABEL_FILE);
            // has 2d array of pixels 0 - 255
            imageList = mnistReader.getImages(TRAINING_IMAGE_FILE);
            dataAmount = 60000;
        }
        else {
            labels = mnistReader.getLabels(TESTING_LABEL_FILE);
            imageList = mnistReader.getImages(TESTING_IMAGE_FILE);
            dataAmount = 10000;
        }

        List<SimpleMatrix[]> data = new ArrayList<>();


        // iterating through each datum
        for (int i = 0; i < dataAmount; ++i) {
            int[] images = arrayDimensionChanger(imageList.get(i));

            SimpleMatrix xMatrix = new SimpleMatrix(pixelAmount, 1);
            SimpleMatrix yMatrix = new SimpleMatrix(outputAmount, 1);

            for (int j = 0; j < pixelAmount; ++j) {
                double valueChanged = images[j] / 255.0;
                xMatrix.set(j, 0, valueChanged);
            }

            int output = labels[i];
            for (int j = 0; j < outputAmount; ++j) {
                if (j == output) {
                    yMatrix.set(j, 0, 1);
                }
            }


            data.add(i, new SimpleMatrix[]{ xMatrix, yMatrix});
        }

        return data;
    }

    // 2d to 1d array
    private int[] arrayDimensionChanger(int[][] inputArray) {
        List<Integer> list = new ArrayList<Integer>();
        for (int i = 0; i < inputArray.length; ++i) {
            for (int j = 0; j < inputArray[i].length; ++j) {
                list.add(inputArray[i][j]);
            }
        }

        int[] outArray = new int[list.size()];
        for (int i = 0; i < list.size(); ++i) {
            outArray[i] = list.get(i);
        }

        return outArray;
    }
}
