package org.smurve.edu.micronn;

import java.util.*;

import static org.smurve.edu.micronn.Functions.*;

public class MicroNNDemo {

    private static final int N_IMAGES = 200;
    private static final int N_TESTS = 20;
    private static final int NUM_EPOCHS = 20;
    private static final double ETA = 0.01;
    private static final double NOISE = 3;

    public static void main(String[] args) {

        new MicroNNDemo().demo();
    }

    /**
     * Here's the actual show going on. Our Network learns to tell diamonds from crosses.
     * A diamant is a 3x3 images similar to
     * 0 1 0
     * 1 0 1
     * 0 1 0
     * <p>
     * with some 10% random added to make it less trivial
     * <p>
     * A cross is a 3x3 image similar to
     * 1 0 1
     * 0 1 0
     * 1 0 1
     * <p>
     * also with some random added
     */
    private void demo() {

        NeuralNetwork nn = new NeuralNetwork(rndMatrix(2, 9), rndVector(2));

        List<LabeledImage> images = getLabeledImages(N_IMAGES);

        System.out.println("Starting training epochs");
        System.out.println("========================");
        for (int n = 0; n < NUM_EPOCHS; n++) {

            double cost = nn.cost(images);
            System.out.println("Cost: " + cost);
            double[][] gradient_m = nn.dC_dm(images);
            double[] gradient_b = nn.dC_db(images);
            nn.M = sub(nn.M, mul(gradient_m, ETA));
            nn.b = sub(nn.b, mul(gradient_b, ETA));
        }
        System.out.println("Done.");


        System.out.println("\nInference against a test set");
        System.out.println("============================");
        List<LabeledImage> testSet = getLabeledImages(N_TESTS);

        for (LabeledImage test : testSet) {
            double[] image = test.image;
            double[] pred_label = nn.f(image);
            Symbol predicted;
            if ( pred_label[0] > pred_label[1] ) {
                predicted = Symbol.DIAMOND;
            } else {
                predicted = Symbol.CROSS;
            }
            printImage(image, predicted);
        }
    }


    private void printImage(double[] image, Symbol predicted) {

        for (int i = 0; i < 3; i++) {
            System.out.printf(" %1.3f  %1.3f  %1.3f\n", image[3 * i], image[3 * i + 1], image[3 * i + 2]);
        }
        System.out.printf("Predicted: %s \n\n", predicted);

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

    /**
     * take an image template and add some random noise
     */
    private double[] rndImage(Symbol symbol) {

        return add(LabeledImage.symbols.get(symbol), mul(rndVector(9), NOISE));
    }

}
