package com.johannesbrodwall.experimental.ml;

import java.util.List;

import no.uib.cipr.matrix.Vector;

public class HandwritingNetwork extends NeuralNetwork {

    public HandwritingNetwork() {
        super(new int[] { 784, 100, 10 });
    }

    public void train(List<Vector> trainingInputs, List<Vector> targetInverseActivations, int trainingSize) {
        super.train(trainingInputs, targetInverseActivations, 30, 10, 3.0, trainingSize);
    }


}
