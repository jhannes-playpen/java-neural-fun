package com.johannesbrodwall.experimental.ml;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Vector;

public class XorNetwork extends NeuralNetwork {

    public XorNetwork() {
        super(new int[] { 2, 2, 1});
    }

    public void construct() {
        weights[1] = new DenseMatrix(new double[][] {{ 54, 14 }, { 17, 14}});
        biases[1] = new DenseVector(new double[] { -8, -20 });
        weights[2] = new DenseMatrix(new double[][] {{ 92, -98 }});
        biases[2] = new DenseVector(new double[] { -48 });
    }

    @Override
    public String toString() {
        String result = new String();

        result += "Input layer: [" + layerSizes[0] + " neurons]\n";
        result += "Hidden layer \n";
        for (int i = 0; i < layerSizes[1]; i++) {
            result += "\tNeuron " + i + " bias: " + biases[1].get(i) + "\n";
        }
        result += " weights:\n" + weights[1];
        result += "Output layer \n";
        for (int i = 0; i < layerSizes[2]; i++) {
            result += "\tNeuron " + i + " bias: " + biases[2].get(i) + "\n";
        }
        result += " weights:\n" + weights[2];

        return result;
    }


    public static void main(String[] args) {
        XorNetwork network = new XorNetwork();
        network.train();
        System.out.println(network);
        network.construct();

        // in0 - 0, in1 - 1
        System.out.println("0,0 " + network.predict(new DenseVector(new double[] { 0, 0 })).get(0));
        System.out.println("0,1 " + network.predict(new DenseVector(new double[] { 0, 1 })).get(0));
        System.out.println("1,0 " + network.predict(new DenseVector(new double[] { 1, 0 })).get(0));
        System.out.println("1,1 " + network.predict(new DenseVector(new double[] { 1, 1 })).get(0));
    }

    private void train() {
        Random random = new Random();

        List<Vector> trainingInputs = new ArrayList<>();
        List<Vector> targetInverseActivations = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            boolean a = random.nextBoolean();
            boolean b = random.nextBoolean();
            trainingInputs.add(new DenseVector(new double[] { a ? 1.0 : 0,  b ? 1.0 : 0 }));
            targetInverseActivations.add(new DenseVector(new double[] { a ^ b ? 0.0 : 1.0 }));
        }
        train(trainingInputs, targetInverseActivations, 10, 10, 3, 80);
    }

    @Override
    protected void evaluate(List<Vector> trainingInputs, List<Vector> targetInverseActivations, int trainingSize) {
        int testSamples = 0;
        int successCount = 0;
        for (int testSample=trainingSize; testSample<trainingInputs.size(); testSample++) {
            boolean expected;
            if (targetInverseActivations.get(testSample).get(0) == 1.0) {
                expected = false;
            } else if (targetInverseActivations.get(testSample).get(0) == 0.0) {
                expected = true;
            } else {
                throw new IllegalArgumentException();
            }
            testSamples++;
            double output = predict(trainingInputs.get(testSample)).get(0);
            if (output > 0.9 && expected) successCount++;
            if (output < 0.9 && !expected) successCount++;
        }
        System.out.println((successCount*100.0)/testSamples + "%");
    }


}
