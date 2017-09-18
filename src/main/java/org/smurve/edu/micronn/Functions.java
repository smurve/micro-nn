package org.smurve.edu.micronn;

import java.util.Arrays;

class Functions {

    enum Symbol {
        DIAMOND,
        CROSS
    }


    /**
     * the neural network function: sigmoid( m*x + b )
     */
    private static double sigmoid(double x) {
        return 1.0 / (1.0 + java.lang.Math.exp(-x));
    }

    static double[] sigmoid(double[] x) {

        return Arrays.stream(x).map(Functions::sigmoid).toArray();
    }

    static double sigmoid_prime(double x) {
        double s = sigmoid(x);
        return s * (1 - s);
    }

    /**
     * multiply the given matrix with the given vector
     *
     * @param m the given matrix
     * @param x the given vector
     * @return the result of the matrix multiplication
     */
     static double[] mul(double[][] m, double[] x) {
        double[] res = new double[m.length];
        for (int i = 0; i < m.length; i++) {
            double sum = 0;
            for (int j = 0; j < m[i].length; j++) {
                sum += m[i][j] * x[j];
            }
            res[i] = sum;
        }
        return res;
    }


    /**
     * multiply matrix with scalar
     */
    static double[][] mul(double[][] m, double c) {
        double[][] res = new double[m.length][];
        for (int i = 0; i < m.length; i++) {
            res[i] = new double[m[0].length];
            for (int j = 0; j < m[0].length; j++) {
                res[i][j] = m[i][j] * c;
            }
        }
        return res;
    }

    /**
     * multiply vector with scalar
     */
    static double[] mul(double[] v, double c) {
        double[] res = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            res[i] = v[i] * c;
        }
        return res;
    }

    /**
     * subtract matrices
     */
    static double[][] sub(double[][] m1, double[][] m2) {
        double[][] res = new double[m1.length][];
        for (int i = 0; i < m1.length; i++) {
            res[i] = new double[m1[0].length];
            for (int j = 0; j < m1[0].length; j++) {
                res[i][j] = m1[i][j] - m2[i][j];
            }
        }
        return res;
    }

    /**
     * add two arbitrary vectors
     */
    static double[] add(double[] x1, double[] x2) {
        assert (x1.length == x2.length);
        double[] res = new double[x1.length];
        for (int j = 0; j < x1.length; j++) {
            res[j] = x1[j] + x2[j];
        }
        return res;
    }

    /**
     * subtract two arbitrary vectors
     */
    static double[] sub(double[] left, double[] right) {
        assert (left.length == right.length);
        double[] res = new double[left.length];
        for (int j = 0; j < left.length; j++) {
            res[j] = left[j] - right[j];
        }
        return res;
    }

    /**
     * l2 norm: the squared difference
     */
    static double l2(double[] x1, double[] x2) {
        assert (x1.length == x2.length);
        double res = 0;
        for (int j = 0; j < x1.length; j++) {
            res += (x1[j] - x2[j]) * (x1[j] - x2[j]);
        }
        return .5 * res;
    }

    /**
     * create a randomly initiated matrix
     *
     * @param nRows number of rows
     * @param nCols number of colums
     * @return a randomly initiated matrix
     */
    static double[][] rndMatrix(int nRows, int nCols) {

        double[][] m = new double[nRows][];
        for (int i = 0; i < nRows; i++) {
            m[i] = new double[nCols];
            for (int j = 0; j < nCols; j++) {
                m[i][j] = (java.lang.Math.random() - 0.5) / 10;
            }
        }
        return m;
    }

    static double[] rndVector(int nCols) {

        double[] res = new double[nCols];
        for (int j = 0; j < nCols; j++) {
            res[j] = (java.lang.Math.random() - 0.5) / 5;
        }
        return res;
    }



}
