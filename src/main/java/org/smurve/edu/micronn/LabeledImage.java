package org.smurve.edu.micronn;

import java.util.HashMap;
import java.util.Map;

class LabeledImage {
    private static final Map<Functions.Symbol, double[]> labels = new HashMap<>();
    static Map<Functions.Symbol, double[]> symbols = new HashMap<>();

    private static final double[] diamond = {
            0, 1, 0,
            1, 0, 1,
            0, 1, 0};

    private static final double[] cross = {
            1, 0, 1,
            0, 1, 0,
            1, 0, 1};


    static {
        symbols.put(Functions.Symbol.DIAMOND, diamond);
        symbols.put(Functions.Symbol.CROSS, cross);
        labels.put(Functions.Symbol.DIAMOND, new double[]{1, 0});
        labels.put(Functions.Symbol.CROSS, new double[]{0, 1});
    }

    double[] image;
    double[] label;

    LabeledImage(double[] image, Functions.Symbol symbol) {
        this.image = image;
        this.label = labels.get(symbol);

    }
}

