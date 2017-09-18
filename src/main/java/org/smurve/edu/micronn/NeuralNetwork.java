package org.smurve.edu.micronn;

import java.util.List;

import static org.smurve.edu.micronn.Functions.*;

class NeuralNetwork {

    double[][] M;
    double[] b;

    NeuralNetwork(double[][] matrix, double[] bias ) {
        this.M = matrix;
        this.b = bias;
    }

    double[] f(double[] x) {
        return sigmoid(
                add(
                        mul(M, x),
                        b
                ));
    }

    /**
     * gradient with respect to the matrix and the entire batch of images
     */
    double[][] dC_dm(List<LabeledImage> images) {

        int cols = images.get(0).image.length;
        int rows = images.get(0).label.length;

        double[][] res = new double[rows][];
        for (int r = 0; r < rows; r++) {
            res[r] = new double[cols];
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double dC_dm_ij = 0;
                for (LabeledImage image : images) {
                    double[] x = image.image;
                    double[] yb = image.label;
                    double deltay_i = sub(f(x), yb)[i];
                    double sigma_prime_i = sigmoid_prime(add(b, mul(M, x))[i]);
                    dC_dm_ij += deltay_i * sigma_prime_i * x[j];
                }
                res[i][j] = dC_dm_ij;
            }
        }

        return res;
    }

    /**
     * gradient with respect to the bias and the entire batch of images
     */
    double[] dC_db(List<LabeledImage> images) {

        int rows = images.get(0).label.length;
        double[] res = new double[rows];
        for (int i = 0; i < rows; i++) {
            double dC_db_i = 0;
            for (LabeledImage image : images) {
                double[] x = image.image;
                double[] yb = image.label;
                double deltay_i = sub(f(x), yb)[i];
                double sigma_prime_i = sigmoid_prime(add(b, mul(M, x))[i]);
                dC_db_i += deltay_i * sigma_prime_i;
            }
            res[i] = dC_db_i;
        }
        return res;
    }

    /**
     * the cost with respect to the given batch
     */
    double cost(List<LabeledImage> images) {
        double cost = 0;
        for (LabeledImage image : images) {
            double[] x = image.image;
            double[] yb = image.label;

            cost += l2(f(x), yb);
        }
        return cost;
    }

}

