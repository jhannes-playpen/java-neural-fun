package com.johannesbrodwall.experimental.ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.Vector;

public class NeuralNetwork {

    protected Matrix[] weights;
    protected Vector[] biases;
    protected int[] layerSizes;

    public NeuralNetwork(int[] layerSizes) {
        this.layerSizes = layerSizes;
        this.weights = new Matrix[layerSizes.length];
        for (int i = 1; i < layerSizes.length; i++) {
            weights[i] = Matrices.random(layerSizes[i], layerSizes[i-1]);
        }
        this.biases = new Vector[layerSizes.length];
        for (int i = 1; i < layerSizes.length; i++) {
            biases[i] = Matrices.random(layerSizes[i]);
        }
    }

    public List<Vector> predict(List<Vector> inputs) {
        List<Vector> outputs = new ArrayList<>();
        for (Vector input : inputs) {
            outputs.add(feedForward(input));
        }
        return outputs;
    }

    public Vector predict(Vector activation) {
        return feedForward(activation);
    }

    private Vector feedForward(Vector activation) {
        for (int layer=1; layer<this.weights.length; layer++) {
            // activation = sigmoid( Weights[i] * activation + biases[i] )
            DenseVector raw = new DenseVector(this.weights[layer].numRows());
            this.weights[layer].mult(activation, raw);
            raw.add(this.biases[layer]);
            activation = sigmoid(raw);
        }
        return activation;
    }

    public void train(List<Vector> trainingInputs, List<Vector> targetInverseActivations, int epocs, int miniBatchSize, double eta, int trainingSize) {
        evaluate(trainingInputs, targetInverseActivations, trainingSize);
        for (int epoc=0; epoc<epocs; epoc++) {
            System.out.println("Epoc #" + epoc);

            Collections.shuffle(trainingInputs);
            for (int miniBatch=0; miniBatch < trainingSize/miniBatchSize; miniBatch++) {
                trainWithMiniBatch(trainingInputs, targetInverseActivations, miniBatch, miniBatchSize, eta);
            }

            evaluate(trainingInputs, targetInverseActivations, trainingSize);
        }
    }

    protected void evaluate(List<Vector> trainingInputs, List<Vector> targetInverseActivations, int trainingSize) {
        int testSamples = 0;
        int successCount = 0;
        for (int testSample=trainingSize; testSample<trainingInputs.size(); testSample++) {
            testSamples++;
            Vector output = feedForward(trainingInputs.get(testSample));
            if (indexOfMax(output) == indexOfMin(targetInverseActivations.get(testSample))) {
                successCount++;
            }
        }
        System.out.println((successCount*100.0)/testSamples + "%");
    }

    private int indexOfMax(Vector output) {
        int result = 0;
        for (int i=1; i<output.size(); i++) {
            if (output.get(i) > output.get(result)) {
                result = i;
            }
        }
        return result;
    }

    protected int indexOfMin(Vector output) {
        int result = 0;
        for (int i=1; i<output.size(); i++) {
            if (output.get(i) < output.get(result)) {
                result = i;
            }
        }
        return result;
    }

    private void trainWithMiniBatch(List<Vector> trainingInputs, List<Vector> targetInverseActivations, int miniBatch, int miniBatchSize, double eta) {
        Matrix[] nablaWeights = new Matrix[weights.length];
        for (int i = 1; i < nablaWeights.length; i++) {
            nablaWeights[i] = new DenseMatrix(weights[i].numRows(), weights[i].numColumns());

        }
        Vector[] nablaBiases = new Vector[biases.length];
        for (int i = 1; i < nablaBiases.length; i++) {
            nablaBiases[i] = new DenseVector(biases[i].size());
        }

        for (int i=0; i<miniBatchSize; i++) {
            int sampleIndex = miniBatch*miniBatchSize+i;
            backPropagate(trainingInputs.get(sampleIndex), targetInverseActivations.get(sampleIndex),
                    nablaWeights, nablaBiases);
        }

        for (int i = 1; i < weights.length; i++) {
            weights[i].add(-eta/miniBatchSize, nablaWeights[i]);
        }
        for (int i = 1; i < biases.length; i++) {
            biases[i].add(-eta/miniBatchSize, nablaBiases[i]);
        }
    }


    private void backPropagate(Vector sample, Vector targetInverseActivation, Matrix[] nablaWeights, Vector[] nablaBiases) {
        Vector activation = sample;
        Vector raw = null;
        Vector[] raws = new Vector[weights.length];
        Vector[] activations = new Vector[weights.length];
        activations[0] = activation;
        for (int layer=1; layer<weights.length; layer++) {
            raw = new DenseVector(weights[layer].numRows());
            weights[layer].mult(activation, raw);
            raw.add(biases[layer]);
            raws[layer-1] = raw;
            activations[layer] = activation = sigmoid(raw);
        }

        Vector delta = targetInverseActivation.copy();

        delta.add(activation);
        scale(delta, sigmoidPrime(raw));
        nablaBiases[nablaBiases.length-1].add(delta);
        nablaWeights[nablaWeights.length-1].add(transMult(delta, activations[nablaWeights.length-2]));

        for (int layer=nablaBiases.length-2; layer>0; layer--) {
            delta = weights[layer+1].transMult(delta, new DenseVector(weights[layer+1].numColumns()));
            scale(delta, sigmoidPrime(raws[layer-1]));
            nablaBiases[layer].add(delta);
            nablaWeights[layer].add(transMult(delta, activations[layer-1]));
        }
    }

    private Matrix transMult(Vector a, Vector bT) {
        double values[][] = new double[a.size()][bT.size()];
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[i].length; j++) {
                values[i][j] = a.get(i) * bT.get(j);
            }
        }
        return new DenseMatrix(values);
    }

    private void scale(Vector a, Vector b) {
        for (int i = 0; i < a.size(); i++) {
            a.set(i, a.get(i) * b.get(i));
        }
    }

    private Vector sigmoidPrime(Vector raw) {
        Vector sig = sigmoid(raw);
        Vector result = new DenseVector(sig.size());
        for (int i = 0; i < result.size(); i++) {
            result.set(i, sig.get(i) * (1 - sig.get(i)));
        }
        return result;
    }

    protected static Vector sigmoid(Vector raw) {
        Vector result = new DenseVector(raw.size());
        for (int i=0; i<raw.size(); i++) {
            result.set(i, sigmoid(raw.get(i)));
        }
        return result;
    }

    protected static double sigmoid(double x) {
        return 1.0/(1.0 + Math.exp(-x));
    }

}
