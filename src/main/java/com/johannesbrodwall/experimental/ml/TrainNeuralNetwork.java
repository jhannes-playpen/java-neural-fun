package com.johannesbrodwall.experimental.ml;

import java.io.IOException;
import java.time.Instant;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import no.uib.cipr.matrix.Vector;

public class TrainNeuralNetwork {

    public static void main(String[] args) throws IOException {
        HandwritingNetwork network = new HandwritingNetwork();

        List<Vector> trainingInputs;
        List<Vector> targetInverseActivations;
        try (MnistDataset dataset = new MnistDataset("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")) {
            trainingInputs = dataset.readImages();
            targetInverseActivations = dataset.readLabelsInverseActivations();
        }

        System.out.println("Training " + Instant.now());
        network.train(trainingInputs, targetInverseActivations, (int) (trainingInputs.size()*0.8));
        System.out.println("Training complete " + Instant.now());

        List<Integer> expected;
        List<Vector> testSamples;
        try (MnistDataset dataset = new MnistDataset("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")) {
            testSamples = dataset.readImages();
            expected = dataset.readLabels();
        }
        List<Vector> actualActivations = network.predict(testSamples);
        System.out.println("Prediction complete " + Instant.now());


        Set<Integer> ok = new HashSet<>();
        Set<Integer> ambigous = new HashSet<>();
        Set<Integer> nonDetermined = new HashSet<>();
        Set<Integer> wrong = new HashSet<>();

        for (int i=0; i<10000; i++) {
            Integer found = null;
            for (int j = 0; j < actualActivations.get(i).size(); j++) {
                double v = actualActivations.get(i).get(j);
                if (v < 0d) continue;
                if (v > .9d) {
                    if (found != null) {
                        ambigous.add(i);
                    } else {
                        found = j;
                    }
                } else {
                    System.out.println("Strange value " + v);
                    System.out.println(actualActivations.get(i).get(j));
                    System.out.println(expected.get(i));
                }
            }
            if (found == null) {
                nonDetermined.add(i);
            } else if (found == expected.get(i)) {
                ok.add(i);
            } else {
                wrong.add(i);
            }

        }

        System.out.println("Prediction analyzed " + Instant.now());

        System.out.println("Ok: " + ok.size() + ", Wrong: " + wrong + ". No answer: " + nonDetermined.size() + ", Multiple answers: " + ambigous.size());



    }

}
