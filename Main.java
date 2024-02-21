import org.ejml.simple.SimpleMatrix;

import java.util.List;

public class Main {

    public static void main(String[] args) {
        TrainingData data = new TrainingData();
        List<SimpleMatrix[]> trainingData = data.getData(true);
        List<SimpleMatrix[]> testingData = data.getData(false);
        int[] layers = {TrainingData.pixelAmount, 30, TrainingData.outputAmount};
        Network network = new Network(layers);
        network.SGD(trainingData, 30, 10, 3, testingData);

        SimpleMatrix output = network.feedForward(trainingData.get(0)[0]);
        System.out.println("Output: " + network.getOutput(output));
        System.out.println("Actual: " + network.getOutput(trainingData.get(0)[1]));
    }

}
