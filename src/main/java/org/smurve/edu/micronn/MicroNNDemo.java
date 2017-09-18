package org.smurve.edu.micronn;

import java.util.*;
import static org.smurve.edu.micronn.Functions.*;

public class MicroNNDemo {

    private static final int N_IMAGES = 200;
    private static final int N_TESTS = 20;
    private static final int NUM_EPOCHS = 10;
    private static final double ETA = 0.01;

    public static void main(String[] args) {

        new MicroNNDemo().demo();
    }

    /**
     * Here's the actual show going on.
     */
    private void demo() {

        NeuralNetwork nn = new NeuralNetwork(rndMatrix(2,9), rndVector(2));

        List<LabeledImage> images = getLabeledImages(N_IMAGES);

        for (int n = 0; n < NUM_EPOCHS; n++) {

            double cost = nn.cost(images);
            System.out.println("Cost: " + cost);
            double[][] gradient_m = nn.dC_dm(images);
            double[] gradient_b = nn.dC_db(images);
            nn.M = sub(nn.M, mul(gradient_m, ETA));
            nn.b = sub(nn.b, mul(gradient_b, ETA));
        }

        List<LabeledImage> testSet = getLabeledImages(N_TESTS);

        for ( LabeledImage test: testSet ) {
            double[] image = test.image;
            double[] given_label = test.label;
            double [] pred_label = nn.f(image);
            System.out.printf("given: (%1.3f, %1.3f) - predicted: (%1.3f, %1.3f)\n", given_label[0], given_label[1], pred_label[0], pred_label[1]);
        }

    }

    private List<LabeledImage> getLabeledImages(int n) {
        List<LabeledImage> images = new ArrayList<>();
        for (int i = 0; i < n / 2; i++) {
            images.add(new LabeledImage(rndImage(Symbol.DIAMOND), Symbol.DIAMOND));
            images.add(new LabeledImage(rndImage(Symbol.CROSS), Symbol.CROSS));
        }
        Collections.shuffle(images);
        return images;
    }

    private double[] rndImage(Symbol symbol) {

        return add(LabeledImage.symbols.get(symbol), rndVector(9));
    }

}
