import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;


public class Network {

    int numLayers;
    int[] sizes;
    SimpleMatrix[] weights;
    SimpleMatrix[] biases;

    public Network(int[] sizes) {
        numLayers = sizes.length;
        this.sizes = sizes;

        Random random = new Random();
        weights = new SimpleMatrix[numLayers - 1];
        biases = new SimpleMatrix[numLayers - 1];
        for (int i = 0; i < numLayers - 1; ++i) {
            weights[i] = SimpleMatrix.random_DDRM(sizes[i], sizes[i + 1], -.2, .2, random);
            biases[i] = SimpleMatrix.random_DDRM(sizes[i + 1], 1, -.2, .2, random);
        }
    }

    // returns output from the network when given input activations
    public SimpleMatrix feedForward(SimpleMatrix activations) {
        for (int i = 0; i < numLayers - 1; ++i) {
            SimpleMatrix z = new SimpleMatrix(sizes[i + 1], 1);
            SimpleMatrix weightCopy = weights[i].copy();
            // finds z value for node j
            for (int j = 0; j < sizes[i + 1]; ++j) {
                // gets all activations from previous layer for node j
                double bias = biases[i].get(j, 0);
                SimpleMatrix weightColumn = weightCopy.extractVector(false, j);
                // j node in i layer
                z.set(j, 0, weightColumn.dot(activations) + bias);
            }
            activations = sigmoid(z);
        }
        return activations;
    }

    // training data: list of each datum, first matrix is input, second is expected output
    // epoch is how many times all of the tests are ran
    // mini batch size is how many tests are done in a batch
    // eta is the small change in variables in backpropogation
    public void SGD(List<SimpleMatrix[]> trainingData, int epochs, int miniBatchSize, double eta,
                    List<SimpleMatrix[]> testData) {
        if (testData != null) {
            int nTest = testData.size();
        }
        int numTests = trainingData.size();

        for (int i = 0; i < epochs; ++i) {
            Collections.shuffle(trainingData);
            List<SimpleMatrix[]>[] miniBatches = new ArrayList[numTests / miniBatchSize];
            for (int j = 0; j  < miniBatches.length; ++j) {
                for (int k = 0; k < miniBatchSize; ++k) {
                    List<SimpleMatrix[]> list = new ArrayList<>();
                    list.add( trainingData.get(j * miniBatchSize + k) );
                    miniBatches[j] = list;
                }
            }

            for (int j = 0; j < miniBatches.length; ++j) {
                updateMiniBatch(miniBatches[j], eta);
            }

            if (testData != null) {
                System.out.println(evaluate(testData) + " / 10000");
            }
            else {
                System.out.println("Loading...");
            }
        }

    }

    // update networks weights and biases through one miniBatch
    // calls backProp to get gradient cost
    private void updateMiniBatch(List<SimpleMatrix[]> miniBatch, double eta) {
        SimpleMatrix[] nablaB = new SimpleMatrix[numLayers - 1];
        SimpleMatrix[] nablaW = new SimpleMatrix[numLayers - 1];
        for (int i = 0; i < numLayers - 1; ++i) {
            nablaB[i] = new SimpleMatrix(sizes[i + 1], 1);
            nablaW[i] = new SimpleMatrix(sizes[i], sizes[i + 1]);
        }

        for (SimpleMatrix[] batch : miniBatch) {
            SimpleMatrix[] deltaNablaB = new SimpleMatrix[numLayers - 1];
            SimpleMatrix[] deltaNablaW = new SimpleMatrix[numLayers - 1];
            for (int i = 0; i < numLayers - 1; ++i) {
                deltaNablaB[i] = new SimpleMatrix(sizes[i + 1], 1);
                deltaNablaW[i] = new SimpleMatrix(sizes[i], sizes[i + 1]);
            }

            // this does all of the work
            backProp( deltaNablaB, deltaNablaW, batch[0], batch[1] );

            for (int j = 0; j < numLayers - 1; ++j) {
                nablaB[j] = nablaB[j].plus(deltaNablaB[j]);
                nablaW[j] = nablaW[j].plus(deltaNablaW[j]);
            }
        }

        for (int i = 0; i < numLayers - 1; ++i) {
            // size / eta is inverse because there is not multiply ):<
            weights[i] = weights[i].minus( nablaW[i].divide(eta / miniBatch.size()) );
            biases[i] = biases[i].minus( nablaB[i].divide(eta / miniBatch.size()) );
        }
    }

    private void backProp(SimpleMatrix[] deltaNablaB, SimpleMatrix[] deltaNablaW, SimpleMatrix x, SimpleMatrix y) {
        SimpleMatrix inputActivation = x.copy();
        SimpleMatrix[] activations = new SimpleMatrix[numLayers];
        activations[0] = inputActivation;
        for (int i = 1; i < numLayers; ++i) {
            activations[i] = new SimpleMatrix(sizes[i], 1);
        }

        // value put into sigmoid for each node: z
        // vectors of z for each layer
        SimpleMatrix[] zs = biases.clone();
        // gets activations and z for layer i
        for (int i = 1; i < numLayers; ++i) {
            // z is a vector for z values in a layer
            SimpleMatrix z = new SimpleMatrix(sizes[i], 1);
            SimpleMatrix weightCopy = weights[i - 1].copy();
            // finds z value for node j
            for (int j = 0; j < sizes[i]; ++j) {
                // gets all activations from previous layer for node j
                double bias = biases[i - 1].get(j, 0);
                SimpleMatrix weightColumn = weightCopy.extractVector(false, j);
                // j node in i layer
                z.set(j, 0, weightColumn.dot(activations[i - 1]) + bias);
            }

            zs[i - 1] = z;
            activations[i] = sigmoid(z);
        }

        // equation #1
        // error for each node in output layer
        SimpleMatrix delta = costDerivative(activations[numLayers - 1], y).elementMult(sigmoidPrime(zs[numLayers - 2]));
        // equation #3
        deltaNablaB[numLayers - 2] = delta.copy();
        // equation #4
        deltaNablaW[numLayers - 2] = activations[numLayers - 2].mult(delta.transpose());

        // indexes get tough to reason out
        for (int i = weights.length - 2; i >= 0; --i) {
            SimpleMatrix z = zs[i];
            SimpleMatrix sp = sigmoidPrime(z);
            delta = weights[i + 1].mult(delta).elementMult(sp);
            deltaNablaB[i] = delta;
            deltaNablaW[i] = activations[i].mult(delta.transpose());
        }
    }

    private SimpleMatrix costDerivative(SimpleMatrix outputActivations, SimpleMatrix y) {
        return outputActivations.minus(y);
    }

    private int evaluate(List<SimpleMatrix[]> testData) {
        int n = 0;
        for (int i = 0; i < testData.size(); ++i) {
           int output = getOutput( feedForward(testData.get(i)[0]) );
           int actual = getOutput( testData.get(i)[1] );
           if (output == actual) {
               ++n;
           }
        }
        return n;
    }

    public int getOutput(SimpleMatrix output) {
        double max = 0;
        int maxPos = -1;
        for (int i = 0; i < TrainingData.outputAmount; ++i) {
            double current = output.get(i, 0);
            if (current > max) {
                max = current;
                maxPos = i;
            }
        }
        return maxPos;
    }

    // UTILITY
    private SimpleMatrix sigmoid(SimpleMatrix z) {
        return z.negative().elementExp().plus(1.0).elementPower(-1);
    }

    private SimpleMatrix sigmoidPrime(SimpleMatrix z) {
        return sigmoid(z).elementMult( sigmoid(z).divide(-1).plus(1) );
    }

}
